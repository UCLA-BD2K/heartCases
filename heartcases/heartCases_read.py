#!/usr/bin/python
#heartCases_read.py
'''
heartCases is a medical language processing system for case reports 
involving cardiovascular disease (CVD).

This part of the system is intended for parsing MEDLINE format files 
and specifically isolating those relevant to CVD using terms in titles 
and in MeSH terms.

Requires bokeh, numpy, nltk, and sklearn.

Uses the Disease Ontology project, the 2017 MeSH Ontology,
and the 2017 SPECIALIST Lexicon. 
The SPECIALIST tools come with their own terms and conditions: see
SPECIALIST.txt.

'''
__author__= "Harry Caufield"
__email__ = "j.harry.caufield@gmail.com"

import argparse, glob, operator, os, random, re, string
import sys, tarfile, time
import urllib, urllib2

from bokeh.layouts import column
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label

import nltk
from nltk.stem.snowball import SnowballStemmer 
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics, preprocessing
from sklearn.externals import joblib

#Constants and Options
					
record_count_cutoff = 30000
	#The maximum number of records to search.
	
random_record_list = False
	#If True, choose training records at random until hitting
	#record_count_cutoff value
	
abst_len_cutoff = 200
	#Abstracts less than this number of characters are considered
	#too short to use for classifier training or re-labeling.
	
train_on_target = True
	#If True, trains the classifier to label only the focus search terms.
	#If False, trains classifier on all terms used in training data.

mesh_topic_tree = { "A07":["025","030","035","037","040","045","231","500","541"],
					"C14":["240", "260","280", "583", "907"],
					"G09":["188","330"]}
	#Lists of codes to include among the MeSH terms used to search
	#documents with (these are domain-specific).
	#Corresponds to MeSH ontology codes.
	#See MN headings in the ontology file.
	#e.g. a heading of C14 and codes of 240 and 280 will include all
	#terms under C14 (Cardiovascular Diseases) and two of the subheadings. 

named_entities = {}
	#Dict of named entities with entity types as keys and sets
	#of terms as values.
	
data_locations = {"do": ("http://ontologies.berkeleybop.org/","doid.obo"),
					"mo": ("ftp://nlmpubs.nlm.nih.gov/online/mesh/MESH_FILES/asciimesh/","d2017.bin"),
					"sl": ("https://lexsrv3.nlm.nih.gov/LexSysGroup/Projects/lexicon/2017/release/LEX/", "LEXICON")}
	#Dict of locations of the external data sets used by this project.

#Classes
class Record(dict):
	'''
	Just establishes that the record is a dict
	'''

#Functions

def find_more_record_text(rec_ids):
	#Retrieves abstract text (or more, if possible)
	#for records lacking abstracts.
	#Takes dict with pmids as keys and pmc_ids as values as input.
	#Returns dict pmids as keys and abstract texts as values.
	
	#Retrieves abstracts from PubMed Central if available.
	#Doesn't search anything else yet.
	
	#This list is processed all at once in order to control
	#the number of requests made through NCBI resources.
	#As this may still involve >500 IDs then NCBI requires
	#call to the History server first as per this link:
	#https://www.ncbi.nlm.nih.gov/books/NBK25499/
	#Note this needs to be a POST request, not a GET
	
	pmc_ids_to_search = []
	newtext_dict = {} #PubMed IDs are keys, abstracts are values
	
	outfilepath = "Additional_PMC_records.txt"
	baseURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
	epost = "epost.fcgi"
	efetch = "efetch.fcgi?db=pmc"
	efetch_options = "&usehistory=y&retmode=text&rettype=medline"
	
	for pmid in rec_ids:
		pmc_id = rec_ids[pmid]
		if pmc_id != "NA": 
			#There is a PubMed Central ID but the article 
			#may be restricted, usually if it's new and embargoed
			
			#Need to truncate first anyway
			pmc_id = pmc_id[3:]
			pmc_ids_to_search.append(pmc_id)
	
	if len(pmc_ids_to_search) > 0:
		print("Searching for %s records in PubMed Central." % len(pmc_ids_to_search))
		
		try:
			#POST using epost first, with all PMC IDs
			idstring = ",".join(pmc_ids_to_search)
			queryURL = baseURL + epost
			args = "?db=pmc" + idstring
			args = urllib.urlencode({"db":"pmc","id":idstring})
			response = urllib2.urlopen(queryURL, args)
			#print(queryURL+args)
			
			response_text = (response.read()).splitlines()
			
			webenv_value = (response_text[3].strip())[8:-9]
			webenv = "&WebEnv=" + webenv_value
			querykey_value = (response_text[2].strip())[10:-11]
			querykey = "&query_key=" + querykey_value
			
			batch_size = 250
			
			i = 0
			while i <= len(pmc_ids_to_search):
				retstart = "&retstart=" + str(i)
				retmax = "&retmax=" + str(i + batch_size)
				queryURL = baseURL + efetch + querykey + webenv \
							+ retstart + retmax + efetch_options
				#print(queryURL)
				response = urllib2.urlopen(queryURL)
				
				out_file = open(outfilepath, "a")
				chunk = 1048576
				while 1:
					data = (response.read(chunk)) #Read one Mb at a time
					out_file.write(data)
					if not data:
						break
					sys.stdout.flush()
					sys.stdout.write(".")
					
				i = i + batch_size 
			out_file.close()
				
			records = medline_parse(open(outfilepath))
			
			for record in records:
				if 'AB' in record.keys() and 'PMID' in record.keys():
					pmid = record['PMID']
					ab = record['AB']
					newtext_dict[pmid] = ab 
	
			print("Retrieved %s new abstracts from PubMed Central." \
					% len(newtext_dict))
			
			return newtext_dict
		
		except urllib2.HTTPError as e:
			print("Couldn't complete PubMed Central search: %s" % e)
	
	else:
		print("No PubMed Central IDs found and/or all records with" \
				" PMC IDs already have abstracts.")
		
def get_data_files(name):
	#Retrieves one of the following:
	#the Disease Ontology database,
	#the 2017 MeSH term file from NLM,
	#or the 2017 SPECIALIST Lexicon from NLM.
	#The last of these requires decompression and returns a directory.
	#The others return a filename.

	baseURL, filename = data_locations[name]
	filepath = baseURL + filename
	outfilepath = filename
		
	print("Downloading from %s" % filepath)
	response = urllib2.urlopen(filepath)
	out_file = open(os.path.basename(filename), "w+b")
	chunk = 1048576
	while 1:
		data = (response.read(chunk)) #Read one Mb at a time
		out_file.write(data)
		if not data:
			print("\n%s file download complete." % filename)
			out_file.close()
			break
		sys.stdout.flush()
		sys.stdout.write(".")
	
	return filename
	
def build_mesh_to_icd10_dict(do_filename):
	#Build the MeSH ID to ICD-10 dictionary
	#the relevant IDs are xrefs in no particular order
	#Also, terms differ in the xrefs provided (e.g., a MeSH but no ICD-10)
	#So, re-assemble the entries first and remove those without both refs
	do_ids = {}	#Internal DOIDs are keys, lists of xrefs are values
				#MeSH ID is always first xref, ICD-10 code is 2nd
				#List also stores parent DOID, if available, in 3rd value
				#4th value is name (a string)
	do_xrefs_icd10 = {} #ICD-10 codes are keys, lists of MeSH IDs are values
						#This isn't ideal but ICD-10 codes are more specific
						#and sometimes the same ICD code applies to multiple 
						#MeSH codes/terms.
	do_xrefs_terms = {} #ICD-10 codes are keys, lists of *all terms* are values
						#Used as fallback in xrefs don't match.
						#As above, one ICD-10 code may match multiple terms,
						#plus these aren't MeSH terms so they may not match
	
	with open(do_filename) as do_file:
		
		have_icd10 = 0
		have_msh = 0
		have_parent = 0
		doid = "0"
		
		#Skip the header on this file
		#The header may vary in line count between file versions.
		for line in do_file:
			if line[0:6] == "[Term]":
				break
			
		for line in do_file:
			if line[0:9] == "id: DOID:":
				doid = ((line.split(":"))[2].strip())
			elif line[0:14] == "xref: ICD10CM:":
				icd10 = ((line.split(":"))[2].strip())
				have_icd10 = 1
			elif line[0:11] == "xref: MESH:":
				msh = ((line.split(":"))[2].strip())
				have_msh = 1
			elif line[0:6] == "is_a: ": #Store parent so we can add it later
				splitline = line.split(":")
				parent = ((splitline[2].split("!"))[0].strip())
				have_parent = 1
			elif line[0:6] == "name: ": #Term name. Does not include synonyms
				name = ((line.split(":"))[1].strip())
			elif line == "\n":
				if have_icd10 == 1 and have_msh == 1:
					if have_parent == 1:
						do_ids[doid] = [msh, icd10, parent, name]
					else:
						do_ids[doid] = [msh, icd10, doid, name]
				else: #We're missing one or both ID refs. Mark the entry
				      #so we can come back to it.
					if have_msh == 1:
						if have_parent == 1:
							do_ids[doid] = [msh, "NA", parent, name]
						else:
							do_ids[doid] = [msh, "NA", doid, name]
					if have_icd10 == 1:
						if have_parent == 1:
							do_ids[doid] = ["NA", icd10, parent, name]
						else:
							do_ids[doid] = ["NA", icd10, doid, name]
					if have_icd10 == 0 and have_msh == 0:
						if have_parent == 1:
							do_ids[doid] = ["NA", "NA", parent, name]
						else:
							do_ids[doid] = ["NA", "NA", doid, name]
				have_icd10 = 0
				have_msh = 0
				have_parent = 0
	
	#Now check to see if we need to inherit refs from parents
	for item in do_ids:
		msh = do_ids[item][0]
		icd10 = do_ids[item][1]
		parent = do_ids[item][2]
		name = do_ids[item][3]
		
		if msh == "NA":
			msh = do_ids[parent][0]
		if icd10 == "NA":
			icd10 = do_ids[parent][1]
			
		if icd10 in do_xrefs_icd10:
			if msh not in do_xrefs_icd10[icd10]:
				do_xrefs_icd10[icd10].append(msh)
		else:
			do_xrefs_icd10[icd10] = [msh]
			
		if icd10 in do_xrefs_terms:
			if name not in do_xrefs_terms[icd10]:
				do_xrefs_terms[icd10].append(name)
		else:
			do_xrefs_terms[icd10] = [name]
	
	return do_ids, do_xrefs_icd10, do_xrefs_terms

def load_slexicon(sl_filename):
	#Loads the terms in the SPECIALIST lexicon.
	#Does not process all fields
	
	lexicon = {}
	
	with open(sl_filename) as sl_file:
		for line in sl_file:
			
			if line[:1] == "{":	#this is a new entry
				base = ""
				cat = ""
				spelling_variant = []
				variants = []
				acronym_of = []
				abbreviation_of = []
				
			if line[:1] == "}": #close entry, add to dict
				base_details = {"cat": cat,
								"spelling_variant": spelling_variant,
								"variants": variants,
								"acronym_of": acronym_of,
								"abbreviation_of": abbreviation_of}
				lexicon[base] = base_details
				continue
				
			splitline = (line.strip()).split("=")
			if splitline[0] == "{base":
				base = splitline[1]
			if splitline[0] == "cat":
				cat = splitline[1]
			if splitline[0] == "spelling_variant":
				spelling_variant.append(splitline[1])
			if splitline[0] == "variants":
				variants.append(splitline[1])
			if splitline[0] == "acronym_of":
				acronym_of.append((splitline[1].split("|"))[0])
			if splitline[0] == "abbreviation_of":
				abbreviation_of.append((splitline[1].split("|"))[0])	
	return lexicon

def build_mesh_dict(mo_filename):
	#Sets up the dict of MeSH terms, specific to the chosen topic.
	#The subset of terms to select (the topic) is defined by global 
	#variable mesh_topic_tree above.
	
	mesh_term_list = [] #All terms to be used for document filtering
						#and term expansion.
						#Note that this does NOT include ALL terms
						#in the MeSH ontology.
	
	mo_ids = {}	#MeSH terms are keys, IDs (UI in term ontology file) 
				#are values
	#Note that this includes ALL terms, not just topic-relevant ones
	#Synonyms get added since a MEDLINE entry may use different terms
	#than the main term we would usually expect
	
	mo_cats = {} #MeSH tree headings (categories) are keys, 
				#values are sets of terms and synonyms
				
	these_synonyms = [] #Full list of synonymous terms for each MeSH term
	
	with open(mo_filename) as mo_file:
		
		for line in mo_file:	#UI is always listed after MH
			if line[0:3] == "MH ": 
				#The main MeSH term (heading)
				term = ((line.split("="))[1].strip()).lower()
				these_synonyms = [term]
			elif line[0:5] == "ENTRY" or line[0:11] == "PRINT ENTRY":
				#Synonymous terms
				entry = (line.split("="))[1].strip()
				synonym = (entry.split("|"))[0].lower()
				these_synonyms.append(synonym)
			elif line[0:3] == "MN ":
				#Location in the MeSH tree. May have multiple locations
				code = (line.split("="))[1].strip()
				codetree = code.split(".")
				tree_cat = codetree[0]
				
				if codetree[0] in mesh_topic_tree:
					if len(codetree) == 1: #This is a category root.
						for synonym in these_synonyms:
							if synonym not in mesh_term_list:
								mesh_term_list.append(synonym)
					elif codetree[1] in mesh_topic_tree[codetree[0]]:  
						for synonym in these_synonyms:
							if synonym not in mesh_term_list:
								mesh_term_list.append(synonym)
				codetree = [""]
			
			#This indicates the end of an entry.
			elif line[0:3] == "UI ":
				clean_id = ((line.split("="))[1].strip())
				mo_ids[term] = clean_id
				
				if tree_cat not in mo_cats:
					mo_cats[tree_cat] = set(term)
				else:
					mo_cats[tree_cat].update([term])
					
				for synonym in these_synonyms:
					mo_ids[synonym] = clean_id
				mo_cats[tree_cat].update(these_synonyms)
				these_synonyms = []
				
		#Ensure there are no duplicates.
		mesh_term_list = set(mesh_term_list)
				
	return mo_ids, mo_cats, mesh_term_list

def get_medline_from_pubmed(pmid_list):
	#Given a file containing a list of PMIDs, one per line.
	#Returns a file containing one MEDLINE record for each PMID.
	outfilepath = pmid_list[:-4] + "_MEDLINE.txt"  
	baseURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed"
	idstring = "&id=" # to be followed by PubMed IDs, comma-delimited
	options = "&retmode=text&rettype=medline"
	
	pmids = []
	
	with open(pmid_list) as infile:
		for line in infile:
			pmids.append(line.rstrip())
	
	print("Retrieving %s records from PubMed." % len(pmids))
	idstring = idstring + ",".join(pmids)
	queryURL = baseURL + idstring + options
	
	response = urllib2.urlopen(queryURL)
	out_file = open(outfilepath, "w+b")
	chunk = 1048576
	while 1:
		data = (response.read(chunk)) #Read one Mb at a time
		out_file.write(data)
		if not data:
			print("\nRecords retrieved - see %s" % outfilepath)
			out_file.close()
			break
		sys.stdout.flush()
		sys.stdout.write(".")
		
	return outfilepath
	
def medline_parse(medline):
	#Parses a MEDLINE file by heading
	#Returns fields out of order, unfortunately
	#Also squishes records if they have the same key value.
	#e.g. all "AD" values end up in one long string
	
	#yields a dict object from a generator object
	
	#These are NOT all the keys in a MEDLINE file,
	#these are just the keys to treat as strings rather than 
	#as lists.
	strkeys = ("ID",
			 "AB",
			 "AD",
			 "CA",
			 "CY",
			 "DA",
			 "DCOM",
			 "DEP",
			 "DP",
			 "EA",
			 "EDAT",
			 "IP",
			 "IS",
			 "JC",
			 "JID",
			 "JT",
			 "LID",
			 "LR",
			 "MHDA",
			 "NI",
			 "OAB",
			 "OWN",
			 "PG",
			 "PL",
			 "PMC",
			 "PMID",
			 "PST",
			 "PUBM",
			 "RF",
			 "SB",
			 "SO",
			 "STAT",
			 "TA",
			 "TI",
			 "TT",
			 "VI",
			 "YR")
			 
	medline = iter(medline)
	key = ""
	record = Record() 
	
	for line in medline:
		line = line.rstrip() 
		if line[:6] == "      ": 
			#A continued field - use whatever the last key was
			if key == "MH": 
				record[key][-1] += line[5:]
			else: 
				record[key].append(line[6:]) #Text starts after the tag
		elif line: 
			#This line is a field
			key = line[:4].rstrip()
			if key not in record:
				record[key] = []
			record[key].append(line[6:])
		elif record: 
			#Finish up the record
			for key in record: 
				if key in strkeys: 
					record[key] = " ".join(record[key]) 
			yield record 
			record = Record() 
			
	if record:
		#Just to take care of the last record 
		for key in record: 
			if key in strkeys: 
				record[key] = " ".join(record[key])
		
		yield record
			
def save_train_or_test_text(msh_terms, title, abst, pmid, cat):
	'''
	Saves the MeSH terms, title, and abstract from a single record
	for training or testing a classifier for both MeSH term
	annotation and sentence topic comprehension.
	
	For the term classifier training:
	 Terms are listed first, separated by |.
	 Appends title to the beginning of the abstract text.
	 Creates folder if not present already.
	 
	'''
	
	if cat == "train":
		tdir = "training_text" #term classifier dir
		
	elif cat == "test":
		tdir = "training_test_text" #term classifier dir

	tfile = "%s.txt" % pmid
	
	if not os.path.isdir(tdir):
		os.mkdir(tdir)
	os.chdir(tdir)
	
	flat_terms = "|".join(msh_terms)
	with open(tfile, 'wb') as outfile:
		outfile.write("%s\t%s %s\t%s" % (flat_terms, title, abst, pmid))
			
	os.chdir("..")

def read_sentence(sentence):
	'''
	Takes a string (usually a sentence) as input.
	Identifies named entities within the string.
	Returns a tuple of:
	* A dict with matched entity types as keys
	  and terms (named entities) as values.
	  If no entities are found then the dict is {"NONE":[]}.
	* and the original string with all named entities highlighted and 
	  tagged with their corresponding
	  types, in the format 
	  I had some ~~chest pain~~[SYMPTOM]
	'''
		
	ne_dict = {} #Keys are entity types, 
				#values are lists of matching terms
					
	#Remove quote marks and hypens
	clean_sentence = ""
	for char in sentence:
		if char not in ["\"", "-"]:
			clean_sentence = clean_sentence + char
	
	#Convert to stems
	cleaner_sentence = clean(clean_sentence)
	
	#Do basic labeling with n-grams where n is 1, 2, or 3
	sent_terms = cleaner_sentence.split()
	
	#Fun list comprehension n-gram approach c/o Scott Triglia; see
	#http://locallyoptimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
	more_sent_terms = []
	for n in [2,3]:
		n_grams = zip(*[sent_terms[i:] for i in range(n)])
		for gram in n_grams:
			more_sent_terms.append(" ".join(gram))
	
	for term in more_sent_terms:
		sent_terms.append(term)
	
	#Convert to a set for efficiency
	sent_terms = set(sent_terms)
	
	for ne_type in named_entities:
		for term in named_entities[ne_type]:
			if term in sent_terms:
				if ne_type not in ne_dict:
					ne_dict[ne_type] = [term]
				else:
					ne_dict[ne_type].append(term)
	if len(ne_dict.keys()) == 0:
		ne_dict["NONE"] = []
		
	for ne_type in ne_dict:
		for term in ne_dict[ne_type]:
			pattern = re.compile(re.escape(term), re.I)
			highlight = "~~\g<0>~~[%s]" % ne_type
			sentence = re.sub(pattern, highlight, sentence)
			
	return (ne_dict, sentence)
		
def parse_training_text(tfile):
	'''
	Loads a MeSH term-labeled title and abstract.
	These are currently stored as one per file.
	Output is a list of MeSH terms, the abstract string, and the PMID.
	'''
	labeled_text = []
	
	for line in tfile: #should only be one line anyway
		splitline = line.split("\t")
		abst = splitline[1].strip()
		msh_terms = (splitline[0]).split("|")
		pmid = splitline[2]
		labeled_text.append([msh_terms, abst, pmid])
	
	return labeled_text

def clean(text):
		'''
		Pre-processing for a string to ensure it lacks stopwords, etc.
		Also removes punctuation.
		Uses NLTK Snowball stemmmer.
		Returns the input string as a processed raw string.
		'''
		stemmer = SnowballStemmer("english")
		stopword_set = set(stopwords.words('english'))
		words = []
		split_text = text.split()
		for word in split_text:
			word = word.decode('utf8') 
			word = word.encode('ascii','ignore')
			#Just to handle Unicode first
				#This may still have an error if there's undecodable
				#Unicode chars, but hopefully those are rare
			word = stemmer.stem(word)
			if word not in stopword_set: #No stopwords
				cleanword = ""
				word = word.lower()
				for char in word:
					if char not in string.punctuation: #No punctuation
						cleanword = cleanword + char
				words.append(cleanword)
		
		cleanstring = " ".join(words)
		return cleanstring

def mesh_classification(testing):
	'''
	Builds and tests a multilabel text classifier for expanding MeSH
	terms used on MEDLINE entries. Uses abstract text as training
	and testing text.
	Uses scikit's DecisionTree implementation.
	
	Looks for a pickled (using joblib) classifier first and uses it 
	if present.
	'''
	
	def load_training_text(no_train):
		
		train_file_list = []
		test_file_list = []
		all_labeled_text = []
		all_test_text = []
		
		if not no_train:
			print("Loading labeled abstracts for training.") 
			train_file_list = glob.glob('training_text/*.txt')
			
			for train_file in train_file_list:
				with open(train_file) as tfile:
					labeled_text = parse_training_text(tfile)
					all_labeled_text = all_labeled_text + labeled_text
			print("Loaded %s labeled abstracts." % len(train_file_list))
		
		if testing:
			print("Loading labeled abstracts for testing.") 
			test_file_list = glob.glob('training_test_text/*.txt')
		
			for test_file in test_file_list:
				with open(test_file) as tfile:
					labeled_text = parse_training_text(tfile)
					all_test_text = all_test_text + labeled_text
			print("Loaded %s labeled abstracts." % len(test_file_list))
		
		return train_file_list, test_file_list, \
				all_labeled_text, all_test_text
	
	clean_labeled_text = []
	all_terms = []
	X_train_pre = [] #Array of text for training
	X_test_pre = [] #Array of text for testing
	y_train = [] #List of lists of labels (MeSH terms) 
	test_labels = [] #List of lists of labels (MeSH terms) in test set
	pmids = [] #List of PMIDs so original records can be retrieved
	
	#This is a multilabel classification so we need a 2D array
	lb = preprocessing.MultiLabelBinarizer()
	
	mesh_term_classifier_name = "mesh_term_classifier.pkl"
	mesh_term_lb_name = "mesh_term_lb.pkl"
	have_classifier = False
	
	if os.path.isfile(mesh_term_classifier_name):
		print("Found previously built MeSH term classifier.")
		have_classifier = True
	
	#Load the input files
	train_file_list, test_file_list, labeled_text, test_text = \
		load_training_text(have_classifier)
	
	if not have_classifier:
		
		t0 = time.time()
		
		print("Learning term dictionary...")
		#Get all terms in use in the training set (the dictionary)
		for item in labeled_text:
			for term in item[0]:
				if term not in all_terms:
					all_terms.append(term)
					
		print("Setting up input...")
		#Set up the input set and term vectors
		for item in labeled_text:
			clean_text = clean(item[1])
			clean_labeled_text.append([item[0], clean_text])
			X_train_pre.append(clean_text)
			y_train.append(item[0])
		
		y_labels = [label for subset in y_train for label in subset]
		y_labels = set(y_labels)
		
		X_train_size = len(X_train_pre)
		
		y_train_bin = lb.fit_transform(y_train)
		
		X_train = np.array(X_train_pre)
	
		#Build the classifier
		print("Building classifier...")
		
		if X_train_size > 10000:
			print("NOTE: This is a large set and may take a while.")
			
		t1 = time.time()
		'''
		This is a scikit-learn pipeline of:
		A vectorizer to extract counts of ngrams, up to 3,
		a tf-idf transformer, and
		a DecisionTreeClassifier, used once per label (OneVsRest) to peform multilabel
		  classification.
		'''
		classifier = Pipeline([
					('vectorizer', CountVectorizer(ngram_range=(1,4), min_df = 5, max_df = 0.5)),
					('tfidf', TfidfTransformer(norm='l2')),
					('clf', OneVsRestClassifier(DecisionTreeClassifier(criterion="entropy",
						class_weight="balanced", max_depth=8), n_jobs=-1))])
		classifier.fit(X_train, y_train_bin)
		
		#Finally, save the classifier and label binarizer
		#(so we can get labels back from vectors in the future)
		print("Saving classifier...")
		joblib.dump(classifier, mesh_term_classifier_name)
		joblib.dump(lb, mesh_term_lb_name)
		
		t2 = time.time()
	
	else:
		print("Loading previous classifier...")
		classifier = joblib.load(mesh_term_classifier_name)
		lb = joblib.load(mesh_term_lb_name)
		
	#Test the classifier
	if testing:
		print("Testing classifier...")
		
		for item in test_text:
			clean_text = clean(item[1])
			X_test_pre.append(clean_text)
			test_labels.append(item[0])
			pmids.append(item[2])
			
		X_test = np.array(X_test_pre)
		
		predicted = classifier.predict(X_test)
		all_labels = lb.inverse_transform(predicted)
		i = 0
		all_recall = [] #To produce average recall value
		all_newlabel_counts = [] #To produce average new label count
		total_new_labels = 0
		
		for item, labels in zip(X_test, all_labels):
			matches = 0
			recall = 0
			new_labels_uniq = [] #Any terms which weren't here before
			new_labels = list(labels)
			for label in new_labels:
				if label in test_labels[i]:
					matches = matches +1
				else:
					new_labels_uniq.append(label)
			recall = matches / float(len(test_labels[i]))
			all_recall.append(recall)
			all_newlabel_counts.append(float(len(new_labels_uniq)))
			i = i+1
	
	t3 = time.time()
	
	if not have_classifier:
		print("\nLoaded dictionary of %s MeSH terms in %.2f seconds." %
				(len(all_terms), (t1 -t0)))
		print("Taught classifier with %s texts in %.2f seconds." %
				(X_train_size, (t2 -t1)))
		print("Overall process required %.2f seconds to complete." %
				((t3 -t0)))
		print("Count of labels used in the training set = %s" % 
				len(y_labels))
	
	if testing:
		avg_recall = np.mean(all_recall)
		avg_newlabel_count = np.mean(all_newlabel_counts)
		print("Average Recall = %s" % avg_recall)
		print("Average new labels added to each test record = %s" % 
				avg_newlabel_count)
	
	return classifier, lb

def plot_those_counts(counts, all_matches, outfilename):
	'''
	Given a dict (with names as values) 
	of dicts of categories and counts,
	produces a simple bar plot of the counts for each.
	Produces plots using Bokeh - produces .html and opens browser.
	'''
	all_plots = []
	
	#Plot simple counts first
	textplot = figure(plot_width=800, plot_height=700, 
					title="Summary Counts")
	textplot.axis.visible = False
	
	i = 100
	for count_name in counts:
		this_count = counts[count_name]
		
		plot_string = "%s: %s" % (count_name, str(this_count))
		
		count_label = Label(x=100, y=i, text=plot_string,
						x_units="screen", y_units="screen",
						border_line_color="white",
						text_font_size="16pt")
		
		textplot.add_layout(count_label)
		
		i = i +100
	
	all_plots.append(textplot)
	
	#Now plot bars for match counts
	for match_name in all_matches:
		matches = all_matches[match_name]
		#Sort categories and make table-like dict
		sort_matches = sorted(matches.items(), key=operator.itemgetter(1),
						reverse=True)
		
		cats = [] #Y axis labels
		catvalues = [] #Y axis
		counts = [] #X axis
		
		randcol = ('#%06X' % random.randint(0,256**3-1)) #A random plot color
	
		#Truncate to top 50 entries
		height = 600
		if len(sort_matches) > 50:
			temp_matches = sort_matches[:50]
			sort_matches = temp_matches
			text_size= "8pt"
		elif len(sort_matches) < 5:
			height = 300
			text_size= "14pt"
		else:
			text_size= "10pt"
	
		i = 0
		for key, value in sort_matches:
			cats.append(key)
			catvalues.append(i)
			counts.append(value)
			i = i +1
			
		match_table = {"Terms": cats, "counts": counts}
		
		plot = figure(plot_width=600, plot_height=height, 
						title=match_name, x_axis_label="Counts")
		plot.hbar(y = catvalues, left=0, right=counts, height=0.5, color=randcol)
		plot.text(counts, catvalues, text=[i for i in cats], 
					text_font_size=text_size, text_baseline="middle")
		
		all_plots.append(plot)
	
	output_file(outfilename, mode="inline", title="heartCases")
	
	#For gridding
	layout = column(all_plots)
	show(layout)
	#show(plot)	

#Main
def main():
	
	record_count = 0 #Total number of records searched, across all files
	match_record_count = 0 #Total number of records matching search terms
							#For now that is keywords in title or MeSH
	abstract_count = 0 #Number of abstracts among the matching records
	rn_counts = 0 #Number of abstracts containing RN material codes
	rn_codes = {} #Keys are RN codes in use, values are counts
	matched_mesh_terms = {} #Keys are terms, values are counts
	matched_journals = {} #Keys are journal titles, values are counts
	matched_years = {} #Keys are years, values are counts
	all_terms_in_matched = {} #Counts of all MeSH terms in matched records
								#Keys are terms, values are counts
	fetch_rec_ids = {} #A dict of record IDs for those records
						#to be searched for additional text retrieval
						#PMIDs are keys, PMC IDs are values
						#or, if no PMC ID available, PMC ID is "NA"
	new_abstract_count = 0 #Number of records with abstracts not directly
							#available through PubMed
	ne_count = 0 #Count of all terms in named_entities
	
	#Set up parser
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputfile', help="name of a text file containing "
						"MEDLINE records")
	parser.add_argument('--pmids', help="name of a text file containing "
						"a list of PubMed IDs to retrieve MEDLINE "
						"records for")
	parser.add_argument('--testing', help="if FALSE, do not test classifiers")
	args = parser.parse_args()
	
	#Get the disease ontology file if it isn't present
	disease_ofile_list = glob.glob('doid.*')
	if len(disease_ofile_list) >1:
		print("Found multiple possible disease ontology files. "
				"Using the preferred one.")
		do_filename = "doid.obo"
	elif len(disease_ofile_list) == 0 :
		print("Did not find disease ontology file. Downloading: ")
		do_filename = get_data_files("do")
	elif len(disease_ofile_list) == 1:
		print("Found disease ontology file: %s " % disease_ofile_list[0])
		do_filename = disease_ofile_list[0]
		
	#Get the MeSH file if it isn't present
	mesh_ofile_list = glob.glob('d2017.*')
	if len(mesh_ofile_list) >1:
		print("Found multiple possible MeSH term files. "
				"Using the preferred one.")
		mo_filename = "d2017.bin"
	elif len(mesh_ofile_list) == 0 :
		print("Did not find MeSH term file. Downloading: ")
		mo_filename = get_data_files("mo")
	elif len(mesh_ofile_list) == 1:
		print("Found MeSH ontology file: %s " % mesh_ofile_list[0])
		mo_filename = mesh_ofile_list[0]
	
	#Get the SPECIALIST lexicon if it isn't present
	sl_file_list = glob.glob('LEXICON*')
	if len(sl_file_list) >1:
		print("Found multiple possible SPECIALIST Lexicon files. "
				"Using the preferred one.")
		sl_filename = "LEXICON"
	elif len(sl_file_list) == 0 :
		print("Did not find SPECIALIST Lexicon file. Downloading: ")
		sl_filename = get_data_files("sl")
	elif len(sl_file_list) == 1:
		print("Found SPECIALIST Lexicon file: %s " % sl_file_list[0])
		sl_filename = sl_file_list[0]
		
	#Now we retrieve MeSH terms so they can be used as IDs
	#AND so terms can be searched for filtering by topic
	#AND so terms can be used for sentence labeling tasks
	
	#Build the MeSH ID dict and the list of MeSH terms specific to the
	#chosen topic. The list includes synonymous terms listed under 
	#ENTRY in the ontology, specific to the chosen topic
	print("Building MeSH ID dictionary and topic-based term list.")
	print("For topic-based terms, using the following MeSH headings "
			"and subheadings:")
	for tree_cat in mesh_topic_tree:
		print("%s (%s)" % (tree_cat, ",".join(mesh_topic_tree[tree_cat])))
		
	mo_ids, mo_cats, mesh_term_list = build_mesh_dict(mo_filename) 
	
	unique_term_count = len(set(mo_ids.values()))
	synonym_count = len(mo_ids) - unique_term_count
	print("Loaded %s unique MeSH terms and %s synonyms "
			"across %s categories." % \
			(unique_term_count, synonym_count, len(mo_cats)))
	
	print("Loaded %s topic-relevant terms + synonyms." % \
			(len(mesh_term_list)))
			
	print("Building MeSH ID to ICD-10 dictionary.")
	#Build the MeSH to ICD-10 dictionary
	do_ids, do_xrefs_icd10, do_xrefs_terms = \
		build_mesh_to_icd10_dict(do_filename)
		
	#Load the SPECIALIST lexicon
	print("Loading SPECIALIST lexicon...")
	slexicon = load_slexicon(sl_filename)
	print("Loaded %s lexicon entries." % len(slexicon))
	
	#Load the named entities here.
	#Most of the vocabulary is inherited from MeSH terms.
	#Clean terms to produce stems.
	print("Loading named entities:")
	
	global named_entities
	
	#Set up the named entity types
	entity_types = []
	
	with open("entity_types.txt") as entity_types_file:
		for line in entity_types_file:
			entity_types.append(line.rstrip())
	
	for ne_type in entity_types:
		named_entities[ne_type] = []
	
	#Populate named entities using the MeSH tree structure
	#as this provides context for terms
	
	for cat in mo_cats:
		if cat in ["D03","D04","D25","D26","D27"]:
			for term in mo_cats[cat]:
				named_entities["drug"].append(term)
		if cat in ["C23"]:
			for term in mo_cats[cat]:
				named_entities["symptom"].append(term)
	
	#Add more terms from other sources at this point...
	#
	#
	#
	
	#Convert list items in named_entities to sets for efficiency.
	for ne_type in named_entities:
		named_entities[ne_type] = set(named_entities[ne_type])
	
	#Then clean up terms and stem them with clean()
	#Ignore some terms we know are not useful for labeling.
	new_named_entities = {}
	
	stop_terms = ["report"]
	
	for ne_type in named_entities:
		new_terms = set()
		for term in named_entities[ne_type]:
			if term not in stop_terms:
				clean_term = clean(term)
				if len(clean_term) > 3:
					new_terms.add(clean_term)
		new_named_entities[ne_type] = new_terms
		
	named_entities = new_named_entities
	
	for ne_type in named_entities:
		ne_count = ne_count + len(named_entities[ne_type])
	ne_type_count = len(named_entities)
	print("Named entity dictionary includes %s terms across %s entity types." % \
			(ne_count, ne_type_count))
	
	#Process CVD-specific MeSH terms produced above
	#These will be used to filter for on-topic documents.
	print("Processing topic-relevant terms.")
	domain_word_list = []
	for term in mesh_term_list:
		domain_word_list.append(clean(term))
	
	#Check if PMID list was provided.
	#If so, download records for all of them.
	if args.pmids:
		pmid_file = str(args.pmids)
		print("Retrieving MEDLINE records for all PMIDs listed in "
				"%s" % pmid_file)
		inputfile = get_medline_from_pubmed(pmid_file)
		medline_file_list = [inputfile]		
	#Check if input file name was provided
	elif args.inputfile:
		medline_file_list = [args.inputfile]
	#otherwise, load all files from input folder
	else:
		medline_file_list = glob.glob('input/*.txt')
		if len(medline_file_list) == 0:
			sys.exit("Found no input files. Exiting.")
		
	ti = 0
	
	matching_orig_records = []
	
	for medline_file in medline_file_list:
		print("\nLoading %s..." % medline_file)
		with open(medline_file) as this_medline_file:
			
			filereccount = 0 #Total number of record in this file

			fileindex = 0 #Index of records in this file (starting at 0)
			
			#Count entries in file first so we know when to stop
			#Could just get length of records but that's too slow
			#So just count entries by PMID

			for line in this_medline_file:
				if line[:6] == "PMID- ":
					filereccount = filereccount +1
			print("\tFile contains %s records." % filereccount)
			
			if filereccount > record_count_cutoff:
				print("\tWill only search %s records." % record_count_cutoff)
			if random_record_list:
				print("Will search randomly.")
			if filereccount == 0:
				sys.exit("No valid entries found in input file.")
			
			this_medline_file.seek(0)
			records = medline_parse(this_medline_file)
			this_medline_file.seek(0)
			
			#Progbar setup
			if filereccount < record_count_cutoff:
				prog_width = filereccount
			else:
				prog_width = record_count_cutoff
			if prog_width < 5000:
				fract = 100
			elif prog_width > 50000:
				fract = 5000
			else:
				fract = 1000
			prog_width = prog_width / fract
			sys.stdout.write("[%s]" % (" " * prog_width))
			sys.stdout.flush()
			sys.stdout.write("\b" * (prog_width+1))
			
			have_records = True
			
			while record_count < record_count_cutoff and fileindex < filereccount:
				
				for record in records:
					
					#If searching randomly, skip some records
					if random_record_list:
						if random.randint(0,1000) > 1:
							fileindex = fileindex +1
							continue
						
					record_count = record_count +1
					
					found = 0
					have_abst = 0
					
					try:
						#Some records don't have titles. Not sure why.
						clean_title = clean(record['TI'])
						split_title = (clean_title).split()
					except KeyError:
						split_title = ["NA"]
					
					for word in domain_word_list:
						if word in split_title:
							found = 1
							break
					
					these_mesh_terms = []
					these_other_terms = []
					try:	
						#If record has no MeSH terms, try the Other Terms
						these_mesh_terms = record['MH']
					except KeyError:
						try:
							these_other_terms = record['OT'] 
							#These need to be in their own list 
							#since they may not work as MeSH terms
						except KeyError:
							these_mesh_terms = []
						
					#Split record MeSH terms into single terms
					#Ignore primary categorization and subcategories
					#This may change later
					#Also list the Other Terms if present
					these_clean_mesh_terms = []
					these_clean_other_terms = []
					these_mesh_codes = []
					these_icd10s = []
					for term in these_mesh_terms:
						clean_term = term.replace("*","")
						clean_term2 = (clean_term.split("/"))[0]
						clean_term3 = (clean_term2.lower())
						these_clean_mesh_terms.append(clean_term3)
						
						these_mesh_codes.append(mo_ids[clean_term3])
					
					for ot in these_other_terms:
						clean_ot = (ot.lower())
						these_clean_other_terms.append(clean_ot)
					
					#Check for matching MeSH terms and other terms
					if found == 0:
						for term in mesh_term_list:
							if term in these_clean_mesh_terms:
								found = 1
								if term not in matched_mesh_terms:
									matched_mesh_terms[term] = 1
								else:
									matched_mesh_terms[term] = matched_mesh_terms[term] +1
							if term in these_clean_other_terms:
								found = 1
					
					if found == 1:
							
						matching_orig_records.append(record)
						
						#Count the journal title
						jtitle = record['JT']
						if jtitle not in matched_journals:
							matched_journals[jtitle] = 1
						else:
							matched_journals[jtitle] = matched_journals[jtitle] +1
							
						#Count the publication year
						pubdate = record['EDAT']
						pubyear = pubdate[:4]
						if pubyear not in matched_years:
							matched_years[pubyear] = 1
						else:
							matched_years[pubyear] = matched_years[pubyear] +1
						
						#Check if there's an abstract
						#and ensure it's not too short - very short
						#abstracts are not informative for the classifier.
						if 'AB' in record.keys() and len(record['AB']) > abst_len_cutoff:
							abstract_count = abstract_count +1
							have_abst = 1
						else: #add IDs for this 
							pmid = record['PMID']
							if 'PMC' in record.keys():
								pmc_id = record['PMC']
								fetch_rec_ids[pmid] = pmc_id
							else:
								fetch_rec_ids[pmid] = "NA"
							
						#Check if there are RN material codes
						#Add them to list if new
						if 'RN' in record.keys():
							rn_counts = rn_counts +1
							for code in record['RN']:
								if code not in rn_codes:
									rn_codes[code] = 1
								else:
									rn_codes[code] = rn_codes[code] +1
						
						match_record_count = match_record_count +1
	
						for term in these_clean_mesh_terms:
							if term not in all_terms_in_matched:
								all_terms_in_matched[term] = 1
							else:
								all_terms_in_matched[term] = all_terms_in_matched[term] +1
						
						'''
						Save 90 percent of matching abstracts to 
						training files in the folder "training_text" and
						10 percent of matching abstracts to testing
						files in the folder "training_test_text".
						 Each file in these folders is one set of 
						 training text from one abstract. Each line of 
						 each file is one set of MeSH terms separated 
						 by |. The terms are followed by a tab and 
						 then the full abstract text.
						 
						 If train_on_target is True,
						 these files are only those containing matching
						 *MeSH terms* specifically and *only* those
						 terms will be used as labels.
						 
						'''
						if train_on_target:
							these_temp_mesh_terms = []
							for term in mesh_term_list:
								if term in these_clean_mesh_terms and term not in these_temp_mesh_terms:
									these_temp_mesh_terms.append(term)
							these_clean_mesh_terms = these_temp_mesh_terms
							if len(these_clean_mesh_terms) == 0:
								have_abst = 0
							
						if have_abst == 1:
							#Doesn't include newly-added abstracts yet
							
							if ti % 10 == 0:
								save_train_or_test_text(these_clean_mesh_terms,
													record['TI'],
													record['AB'],
													record['PMID'],
													"test")
							else:
								save_train_or_test_text(these_clean_mesh_terms, 
													record['TI'],
													record['AB'],
													record['PMID'],
													"train")
							ti = ti+1
					
					fileindex = fileindex +1
					
					sys.stdout.flush()
					if record_count % fract == 0:
						sys.stdout.write("#")
					if record_count == record_count_cutoff:
						break
	
	have_new_abstracts = False
	
	if len(fetch_rec_ids) > 0:
		print("\nFinding additional abstract text for records.")
		new_abstracts = find_more_record_text(fetch_rec_ids)
		try:
			new_abstract_count = len(new_abstracts)
			have_new_abstracts = True
		except TypeError:
			print("Found no new abstracts.")

	#Really, new abstracts should get added earlier so they can be used
	#as part of the classifier, but mostly we need them for
	#term expansion
	
	print("\nTotal input contains:\n"
			"%s records,\n"
			"%s matching records,\n"
			"%s matching records with abstracts\n"
			"(%s abstracts retrieved from additional sources.)"
			% (record_count, match_record_count, abstract_count,
				new_abstract_count))
	
	#MeSH terms are often incomplete, so here they are used
	#to train a classifier and identify associations
	#which can then be used to extend the existing annotations.
	if train_on_target:
		print("\nStarting to build term classifier with target "
				"terms only...")
	else:
		print("\nStarting to build term classifier for all terms "
				"used in the training abstracts...")
	
	#Argument tells us if we should not test the classifier
	#This saves some time.
	testing = True
	if args.testing:
		if args.testing == "FALSE":
			testing = False

	abst_classifier, lb = mesh_classification(testing)
	#Also returns the label binarizer, lb
	#So labels can be reproduced
	
	#This is the point where we need to add the new MeSH terms
	#and *then* search for new matching ICD-10 codes
	#Denote newly-added terms with ^

	matching_ann_records = []

	print("\nAdding new terms and codes to records.")
	
	#Progbar setup
	prog_width = len(matching_orig_records) 
	if prog_width < 5000:
		prog_width = prog_width / 100
		shortbar = True
	else:
		prog_width = prog_width / 1000
		shortbar = False
	sys.stdout.write("[%s]" % (" " * prog_width))
	sys.stdout.flush()
	sys.stdout.write("\b" * (prog_width+1))
	j = 0
	
	for record in matching_orig_records:
		'''
		Use classifier on abstract to get new MeSH terms,
		append new terms to record["MH"], and
		add new ICD-10 codes.
		Ensure new terms are denoted properly as above.	
		This step can be very slow - 
		mostly due to classifier predictions.
		'''
		
		these_mesh_terms = []
		these_other_terms = []
		these_clean_mesh_terms = []
		these_clean_other_terms = []
		these_mesh_codes = []
		these_icd10s = []
		
		have_more_terms = False
		
		#We may have an abstract retrieved from somewhere else
		#if so, add it first
		if 'AB' not in record.keys() and have_new_abstracts:
			if record['PMID'] in new_abstracts:
				record['AB'] = new_abstracts[record['PMID']]
		
		if 'AB' in record.keys() and len(record['AB']) > abst_len_cutoff:
			titlestring = record['TI']
			abstring = record['AB']
			clean_string = "%s %s" % (clean(titlestring), clean(abstring))
			clean_array = np.array([clean_string])
			predicted = abst_classifier.predict(clean_array)
			all_labels = lb.inverse_transform(predicted)
			have_more_terms = True

		try:	
			#If record has no MeSH terms, try the Other Terms
			these_mesh_terms = record['MH']
		except KeyError:
			try:
				these_mesh_terms = []
				these_other_terms = record['OT'] 
				#These need to be in their own list 
				#since they may not work as MeSH terms
				#they'll be ignored anyway if they don't resolve
				#to MeSH codes for cross-referencing
			except KeyError:
				these_mesh_terms = []
		
		for term in these_mesh_terms:
			clean_term = term.replace("*","")
			clean_term2 = (clean_term.split("/"))[0]
			clean_term3 = (clean_term2.lower())
			these_clean_mesh_terms.append(clean_term3)
			these_mesh_codes.append(mo_ids[clean_term3])
		
		these_clean_mesh_terms = set(these_clean_mesh_terms)
		
		if have_more_terms:
			for term in all_labels[0]: #Technically a tuple in a list.
				clean_term = (term.lower())
				if clean_term not in these_clean_mesh_terms:
					if clean_term != '':
						try:
							record['MH'].append("%s^" % term)
						except KeyError:
							record['MH'] = ["%s^" % term]
						these_mesh_codes.append("%s^" % mo_ids[clean_term])
					
		these_mesh_codes = set(these_mesh_codes)
				
		#Find and add the new ICD-10 codes using ID x-refs
		
		for msh in these_mesh_codes:
			predicted = False
			if msh[-1:] == "^": #This means it's predicted
				predicted = True
				msh = msh[:-1]
			for code in do_xrefs_icd10:
				if msh in do_xrefs_icd10[code] and code != "NA":
					if code not in these_icd10s:
						if predicted == True:
							these_icd10s.append("%s^" % code)
						else:
							these_icd10s.append(code)
					break
		
		#Add more ICD-10 codes based off terms alone		
		for mshterm in these_clean_mesh_terms:
			for code in do_xrefs_terms:
				if mshterm in do_xrefs_terms[code] and code != "NA":
					if code not in these_icd10s and \
						("%s^" % code) not in these_icd10s:
						if predicted == True:
							these_icd10s.append("%s^" % code)
						else:
							these_icd10s.append(code)
					break
						
		for code in these_icd10s:
			#Record may not have OT (other terms) section yet
			try:
				record['OT'].append("ICD10CM:%s" % code)
			except KeyError:
				record['OT'] = [("ICD10CM:%s" % code)]
		
				
		matching_ann_records.append(record)
		
		j = j+1
		
		sys.stdout.flush()
		if shortbar:
			if j % 100 == 0:
				sys.stdout.write("#")
		else:
			if j % 1000 == 0:
				sys.stdout.write("#")
	
	'''
	Output the matching entries, complete with new annotations
	Note that, unlike in original MEDLINE record files,
	this output does not always place long strings on new lines.
	'''
	
	#Now abstract text is searched for named entities
	
	if not os.path.isdir("output"):
		os.mkdir("output")
	os.chdir("output")
	
	if match_record_count < record_count_cutoff:
		filelabel = match_record_count
	else:
		filelabel = record_count_cutoff
		
	if len(medline_file_list) == 1:
		outfilename = (medline_file_list[0])[6:-4] + "_%s_out.txt" \
						% filelabel
		raw_ne_outfilename = (medline_file_list[0])[6:-4] + "_%s_raw_ne.txt" \
						% filelabel
		labeled_outfilename = (medline_file_list[0])[6:-4] + "_%s_labeled.txt" \
						% filelabel
		viz_outfilename = (medline_file_list[0])[6:-4] + "_%s_plots.html" \
						% filelabel
	else:
		outfilename = "medline_entries_%s_out.txt" \
						% filelabel
		raw_ne_outfilename = "medline_entries_%s_raw_ne.txt" \
						% filelabel
		labeled_outfilename = "medline_entries_%s_labeled.txt" \
						% filelabel
		viz_outfilename = "medline_entries_%s_plots.html" \
						% filelabel 
	
	print("\nWriting matching, newly annotated full records to file...")
	
	with open(outfilename, 'w') as outfile:
		for record in matching_ann_records:
			for field in record:
				if len(field) == 4:
					formfield = field
				elif len(field) == 3:
					formfield = "%s " % field
				elif len(field) == 2:
					formfield = "%s  " % field
				
				#List items get one line per item
				if isinstance(record[field], list):
					for value in record[field]:
						outfile.write("%s- %s\n" % (formfield, value))
				else:
					#Some strings, like abstracts and sometimes titles,
					#are often too long to be human readable 
					#so they get broken up
					if field in ['AB', 'OAB', 'TI']:
						newform = []
						charcount = 0
						splitstr = record[field].split(" ")
						i = 0
						for word in splitstr:
							i = i +1
							charcount = charcount + len(word) +1
							if charcount >= 70:
								if i == len(splitstr):
									word = word
								else:
									word = "%s\n     " % word
									charcount = 0
							newform.append(word)
		
						outfile.write("%s- %s\n" % (formfield, " ".join(newform)))
					else:
						outfile.write("%s- %s\n" % (formfield, record[field]))
					
			outfile.write("\n")
	
	#Labeling sentences within the matching records
	#using the read_sentence function
	labeled_ann_records = []
	for record in matching_ann_records:
		labeled_record = {}
		labeled_record['TI'] = "NO TITLE"
		labeled_record['labstract'] = [["NO ABSTRACT",["NONE"]]]
		for field in record:
			if field == 'TI':
				labeled_record['TI'] = record[field]
			if field == 'PMID':
				labeled_record['PMID'] = record[field]
			if field == 'AB':
				abstract = record[field]
				labeled_abstract = []
				try:
					sentence_list = sent_tokenize(abstract)
				except UnicodeDecodeError:
					abstract = abstract.decode('utf8')
					abstract = abstract.encode('ascii','ignore')
					sentence_list = sent_tokenize(abstract)
					
				for sentence in sentence_list:
					lsentence = read_sentence(sentence)
					labeled_abstract.append(lsentence)
				
				labeled_record['labstract'] = labeled_abstract
			
		labeled_ann_records.append(labeled_record)
			
	#Writing abstract with labeled entities to files
	
	print("\nWriting raw entities found in abstracts...")
	
	with open(raw_ne_outfilename, 'w') as outfile:
		for record in labeled_ann_records:
			
			outfile.write("%s\n" % record['TI'])
			outfile.write("%s\n" % record['PMID'])
			for lsentence in record['labstract']:
				ne = str(lsentence[0])
				outfile.write("%s\n" % ne)
			
			outfile.write("\n\n")
			
	print("\nWriting text with labeled entities for matching records...")
	
	with open(labeled_outfilename, 'w') as outfile:
		for record in labeled_ann_records:
			
			outfile.write("%s\n" % record['TI'])
			outfile.write("%s\n" % record['PMID'])
			for lsentence in record['labstract']:
				sentence = lsentence[1]
				outfile.write("%s\n" % sentence)
			
			outfile.write("\n\n")
	
	if record_count > 0:
		
		#Plot first.
		#Then provide some summary statistics.
		counts = {"Total searched records": record_count,
					"Records with matched term in the title or MeSH term": match_record_count,
					"Number of records with abstracts": abstract_count,
					"Number of records with material codes": rn_counts,
					"Total unique material codes used": len(rn_codes)}
		all_matches = {"Most frequently matched MeSH terms": matched_mesh_terms,
						"Most common MeSH terms in matched records": all_terms_in_matched,
						"Most records are from these journals": matched_journals, 
						"Most records were published in these years": matched_years,
						"Most frequently used material codes": rn_codes}
						
		plot_those_counts(counts, all_matches, viz_outfilename)
		
		all_matches_high = {}
		for entry in all_matches:
			all_matches_high[entry] = sorted(all_matches[entry].items(), key=operator.itemgetter(1),
									reverse=True)[0:15]
		
		for entry in counts:
			print("\n%s:\n %s" % (entry, counts[entry]))
		
		for entry in all_matches_high:
			print("\n%s:\n" % entry)
			for match_count in all_matches_high[entry]:
				print("%s\t\t%s" % (match_count[0], match_count[1]))
				
	else:
		sys.exit("Found no matching references.")
	
	print("\nDone - see the following files in the output folder:\n"
			"%s for the full matching records with MEDLINE headings,\n"
			"%s for raw entities found in abstracts,\n"
			"%s for entity-labeled abstract sentences, and\n"
			"%s for plots." %
			(outfilename, raw_ne_outfilename, 
			labeled_outfilename, viz_outfilename))
	
if __name__ == "__main__":
	sys.exit(main())
