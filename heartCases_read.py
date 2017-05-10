#!/usr/bin/python
#heartCases_read.py
'''
heartCases is a medical language processing system for case reports 
involving cardiovascular disease (CVD).

This part of the system is intended for parsing MEDLINE format files 
and specifically isolating those relevant to CVD using terms in titles 
and in MeSH terms.

Requires numpy, nltk, and sklearn.

'''
__author__= "Harry Caufield"
__email__ = "j.harry.caufield@gmail.com"

import argparse, glob, operator, os, random, re, string, sys, time
import urllib, urllib2

import nltk
from nltk.stem.snowball import SnowballStemmer 
'''
Testing this one due to recent bugs in Porter stemmer
Otherwise use this:
'''
#from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.svm import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics, preprocessing
from sklearn.externals import joblib


#Constants and Options
					
record_count_cutoff = 1000
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

mesh_topic_heading = ["C14"]
mesh_topic_tree = ["240", "260","280", "583", "907"]
	#Lists of codes to include among the MeSH terms used.
	#Corresponds to MeSH ontology codes.
	#See MN headings in the ontology file.
	#e.g. a heading of C14 and codes of 240 and 280 will include all
	#terms under C14 (Cardiovascular Diseases) and two of the subheadings. 

sentence_label_filename = "sentence_label_terms.txt"
	#Contains of labels and vocabulary used to pre-classify sentences.

sentence_labels = {}
	#Dict of sentence labels with sets of associated terms as keys.
	#Expanded later

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
		
	
def get_heart_words(): #Reads a file of topic-specific vocabulary
	#In this case, the topic is heart disease.
	word_list = []
	with open("heart_dis_vocab.txt") as heart_word_file:
		for line in heart_word_file:
			word_list.append(line.rstrip())
	return word_list
	
def get_disease_ontology(): #Retrieves the Disease Ontology database
	baseURL = "http://ontologies.berkeleybop.org/"
	ofilename = "doid.obo"
	ofilepath = baseURL + ofilename
	outfilepath = ofilename
	
	print("Downloading from %s" % ofilepath)
	
	response = urllib2.urlopen(ofilepath)
	out_file = open(os.path.basename(ofilename), "w+b")
	chunk = 1048576
	while 1:
		data = (response.read(chunk)) #Read one Mb at a time
		out_file.write(data)
		if not data:
			print("\n%s file download complete." % ofilename)
			out_file.close()
			break
		sys.stdout.flush()
		sys.stdout.write(".")
		
	return ofilename
	
def get_mesh_ontology(): #Retrieves the 2017 MeSH term file from NLM
	baseURL = "ftp://nlmpubs.nlm.nih.gov/online/mesh/MESH_FILES/asciimesh/"
	mfilename = "d2017.bin"
	mfilepath = baseURL + mfilename
	outfilepath = mfilename
	
	print("Downloading from %s" % mfilepath)
	
	response = urllib2.urlopen(mfilepath)
	out_file = open(os.path.basename(mfilename), "w+b")
	chunk = 1048576
	while 1:
		data = (response.read(chunk)) #Read one Mb at a time
		out_file.write(data)
		if not data:
			print("\n%s file download complete." % mfilename)
			out_file.close()
			break
		sys.stdout.flush()
		sys.stdout.write(".")
		
	return mfilename

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

def build_mesh_dict(mo_filename):
	#Sets up the dict of MeSH terms, specific to the chosen topic.
	
	#The subset of terms to select (the topic) is defined by global 
	#variable mesh_topic_tree above
	mesh_term_list = []
	
	mo_ids = {}	#MeSH terms are keys, IDs (UI in term ontology file) 
				#are values
	#Note that this includes ALL terms, not just topic-relevant ones
	#Synonyms get added since a MEDLINE entry may use different terms
	#than the main term we would usually expect
	
	mo_cats = {} #MeSH tree headings (categories) are keys, 
				#values are sets of terms and synonyms
				
	these_synonyms = [] #Synonymous terms for teach MeSH term
	
	with open(mo_filename) as mo_file:
		
		for line in mo_file:	#UI is always listed after MH
			if line[0:3] == "MH ":
				term = ((line.split("="))[1].strip()).lower()
				these_synonyms = [term]
			elif line[0:5] == "ENTRY" or line[0:11] == "PRINT ENTRY":
				entry = (line.split("="))[1].strip()
				synonym = (entry.split("|"))[0].lower()
				these_synonyms.append(synonym)
			elif line[0:3] == "MN ":
				code = (line.split("="))[1].strip()
				codetree = code.split(".")
				tree_cat = codetree[0]
				
				if codetree[0] in mesh_topic_heading:
					if len(codetree) == 1:
						for synonym in these_synonyms:
							if synonym not in mesh_term_list:
								mesh_term_list.append(synonym)
					elif codetree[1] in mesh_topic_tree: 
						#This will select term subsets 
						for synonym in these_synonyms:
							if synonym not in mesh_term_list:
								mesh_term_list.append(synonym)
				codetree = ""
			
			#this indicates the end of an entry.
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
	 
	For the sentence topic classifier training:
	 Calls label_sentence() on each sentence to do pre-labeling.
	 Saves labeled sentences to a comma-delimited file of two columns.
	 One or more labels, separated by |, are in the first column.
	 A single sentence per row is in the second column.
	 The sentence begins and end with double quotation marks.
	 Any existing quotation marks are removed from the sentence.
	 Unlike the term classifier, this produces only two files,
	 one for both testing and training.
	 
	 Returns count of all training sentences (training + testing).
	'''
	
	if cat == "train":
		tdir = "training_text" #term classifier dir
		sdir = "training_sentences" #sentence classifier dir
		sentence_filename = "training_sentences.txt"
		
	elif cat == "test":
		tdir = "training_test_text" #term classifier dir
		sdir = "training_test_sentences" #sentence classifier dir
		sentence_filename = "testing_sentences.txt"
		
	tfile = "%s.txt" % pmid
	
	if not os.path.isdir(tdir):
		#print("Setting up MeSH term classifier %sing directory." % cat)
		os.mkdir(tdir)
	os.chdir(tdir)
	
	flat_terms = "|".join(msh_terms)
	with open(tfile, 'wb') as outfile:
		outfile.write("%s\t%s %s\t%s" % (flat_terms, title, abst, pmid))
			
	os.chdir("..")
	
	#Now append sentences to the sentence topic training file
	#and do some basic pre-labeling with terms we know should be 
	#associated with labels (performed by label_sentence).
	sentence_count = 0
	
	#print("**DOCUMENT: %s**" % title)
	
	if not os.path.isdir(sdir):
		#print("Setting up sentence classifier %sing directory." % cat)
		os.mkdir(sdir)
	os.chdir(sdir)
	
	sentence_file = open(sentence_filename, "a")
	
	for sentence in sent_tokenize(abst):
		topic_list = label_sentence(sentence)
		topic = "|".join(topic_list)
		sentence_file.write("%s,\"%s\"\n" % (topic, sentence))
		sentence_count = sentence_count +1
	
	os.chdir("..")
	
	return sentence_count

def label_sentence(sentence):
	#Takes a string (usually of multiple words) as input.
	#Labels the string using labels and terms in sentence_labels.
	#Returns a list of matching labels.
	
	#Eventually will also extract numerical values, e.g. lab results.
	
	label_list = []
					
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
	
	for label in sentence_labels:
		for term in sentence_labels[label]:
			if len(term) > 2 \
				and term not in ["parent","patient"] \
				and term in sent_terms \
				and label not in label_list:
				label_list.append(label)
				#print("%s = [%s] in %s" % (label, term, sentence))
				break #May not actually want to break if we want greedy matches
	if len(label_list) == 0:
		label_list.append("NONE")
	
	return label_list

def parse_training_text(tfile):
	'''
	Loads a MeSH term-labeled title and abstract.
	These are currently stored as one per file.
	Output is a list of MeSH terms, the abstract string, and the PMID.
	'''
	labeled_text = []
	
	#print("Reading %s" % (tfile.name))
	for line in tfile: #should only be one line anyway
		splitline = line.split("\t")
		abst = splitline[1].strip()
		msh_terms = (splitline[0]).split("|")
		pmid = splitline[2]
		labeled_text.append([msh_terms, abst, pmid])
	
	return labeled_text

def parse_training_sentences(sfile):
	'''
	Loads a file of labelled sentences for classifier training.
	One file may contain numerous sentences, one per line,
	with the sentence text in double quotes.
	Output is a list of labels and the sentence text.
	'''
	labeled_sentences = []
	
	for line in sfile:
		labels = []
		for label in ((line.split(","))[0]).split("|"):
			labels.append(label)
		text = ((line.split("\"", 1))[1])[:-1]
			#Split at the first quote mark, get the text after it,
			#and remove the final quote mark.
		labeled_sentences.append([labels, text])
	
	return labeled_sentences

def clean(text):
		'''
		Pre-processing for a string to ensure it lacks stopwords, etc.
		Also removes punctuation.
		Uses NLTK Snowball stemmmer which is essentially the Porter stemmer.
		Returns the input string as a processed raw string.
		'''
		#stemmer = PorterStemmer()
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
			#print '%s => %s' % (item, '|'.join(labels))
			new_labels = list(labels)
			for label in new_labels:
				if label in test_labels[i]:
					matches = matches +1
				else:
					new_labels_uniq.append(label)
			recall = matches / float(len(test_labels[i]))
			all_recall.append(recall)
			#print("Recall: %s" % recall)
			#print("New Terms: %s" % ("|".join(new_labels_uniq)))
			all_newlabel_counts.append(float(len(new_labels_uniq)))
			#print(pmids[i])
			i = i+1
	
	t3 = time.time()
	
	if not have_classifier:
		print("\nLoaded dictionary of %s terms in %.2f seconds." %
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

def sent_classification(testing):
	'''
	Builds and tests a multilabel text classifier for labelling
	sentences from abstracts using one of 12 labels. 
	Uses abstract text, tokenized by sentence and pre-labelled using 
	general terms (i.e. those in sentence_label_terms.txt) and presence 
	of named entities from MeSH headings.
	Uses scikit's DecisionTree implementation.
	
	Most of the work is already done in the pre-labelling step,
	so this classifier is primarily for verification.
	
	The classifier may add NONE as its own label though this is
	a unique label class denoting no match. This is useful as a
	diagnostic so it is kept at this stage but removed before
	producing the output file.
	
	Looks for a pickled (using joblib) classifier first and uses it 
	if present.
	'''
	
	def load_training_sents(no_train):
		
		all_labeled_text = []
		all_test_text = []
		
		if not no_train:
			print("Loading labeled sentences for training.") 
			
			with open('training_sentences/training_sentences.txt') as sfile:
				all_labeled_text = parse_training_sentences(sfile)
			print("Loaded %s labeled sentences." % len(all_labeled_text))
		
		if testing:
			print("Loading labeled sentences for testing.") 
			with open('training_test_sentences/testing_sentences.txt') as sfile:
				all_test_text = parse_training_sentences(sfile)
			print("Loaded %s labeled sentences." % len(all_test_text))
		
		return all_labeled_text, all_test_text
	
	clean_labeled_text = []
	all_terms = []
	X_train_pre = [] #Array of text for training
	X_test_pre = [] #Array of text for testing
	y_train = [] #List of lists of labels
	test_labels = [] #List of lists of labels in test set
	
	#This is a multilabel classification so we need a 2D array
	lb = preprocessing.MultiLabelBinarizer()
	
	sentence_classifier_name = "sentence_label_classifier.pkl"
	sentence_classifier_lb_name = "sentence_label_lb.pkl"
	have_classifier = False
	
	if os.path.isfile(sentence_classifier_name):
		print("Found previously built sentence label classifier.")
		have_classifier = True
	
	#Load the input files
	labeled_text, test_text = load_training_sents(have_classifier)
	
	if testing and len(test_text) == 0:
		sys.exit("Didn't load any training sentences - " \
					"please verify they are present and try again.")
	
	if not have_classifier:
		
		t0 = time.time()
		
		#Load sentence labels
		for label in sentence_labels:
			all_terms.append(label)
					
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
			
		t1 = time.time()
		'''
		This is a scikit-learn pipeline of:
		A vectorizer to extract counts of ngrams, up to 2,
		a tf-idf transformer, and
		a DecisionTreeClassifier, used once per label (OneVsRest) to peform multilabel
		  classification.
		'''
		classifier = Pipeline([
					('vectorizer', CountVectorizer(ngram_range=(1,3), min_df = 5, max_df = 0.5)),
					('tfidf', TfidfTransformer(norm='l2')),
					('clf', OneVsRestClassifier(SGDClassifier(loss="perceptron", eta0=1,
						learning_rate="constant", penalty='l1', n_jobs=4)))])
		classifier.fit(X_train, y_train_bin)
		
		#Finally, save the classifier
		print("Saving classifier...")
		joblib.dump(classifier, sentence_classifier_name)
		joblib.dump(lb, sentence_classifier_lb_name)
		
		t2 = time.time()
	
	else:
		classifier = joblib.load(sentence_classifier_name)
		lb = joblib.load(sentence_classifier_lb_name)
		
	#Test the classifier
	if testing:
		print("Testing sentence label classifier...")
		
		for item in test_text:
			clean_text = clean(item[1])
			X_test_pre.append(clean_text)
			test_labels.append(item[0])
			
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
			#print '%s => %s' % (item, '|'.join(labels))
			new_labels = list(labels)
			for label in new_labels:
				if label in test_labels[i]:
					matches = matches +1
				else:
					new_labels_uniq.append(label)
			recall = matches / float(len(test_labels[i]))
			all_recall.append(recall)
			#print("Recall: %s" % recall)
			#print("New Terms: %s" % ("|".join(new_labels_uniq)))
			all_newlabel_counts.append(float(len(new_labels_uniq)))
			i = i+1
	
	t3 = time.time()
	
	if not have_classifier:
		print("\nLoaded dictionary of %s terms in %.2f seconds." %
				(len(all_terms), (t1 -t0)))
		print("Taught classifier with %s sentences in %.2f seconds." %
				(X_train_size, (t2 -t1)))
		print("Overall process required %.2f seconds to complete." %
				((t3 -t0)))
		print("Count of labels (including NONE) used in the " \
				"training set = %s" % len(y_labels))
	
	if testing:
		avg_recall = np.mean(all_recall)
		avg_newlabel_count = np.mean(all_newlabel_counts)
		print("Average Recall = %s" % avg_recall)
		print("Average new labels added to each test record = %s" % 
				avg_newlabel_count)
	
	return classifier, lb
	
#Main
def main():
	
	record_count = 0 #Total number of records searched
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
	sent_count = 0 #Number of sentences saved for training a sentence
					#classifier
	label_term_count = 0 #Count of all terms in sentence_labels
	
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
		do_filename = get_disease_ontology()
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
		mo_filename = get_mesh_ontology()
	elif len(mesh_ofile_list) == 1:
		print("Found MeSH ontology file: %s " % mesh_ofile_list[0])
		mo_filename = mesh_ofile_list[0]
	
	#Retrieves the list of topic-specific words we're interested in
	heart_word_list = []
	raw_heart_word_list = get_heart_words()
	for word in raw_heart_word_list:
		heart_word_list.append(clean(word))
	print("Loaded %s topic-specific words." % len(heart_word_list))
	
	#Now we retrieve MeSH terms so they can be used as IDs
	#AND so terms can be searched for filtering by topic
	#AND so terms can be used for sentence labeling tasks
	
	#Build the MeSH ID dict and the list of MeSH terms specific to the
	#chosen topic. The list includes synonymous terms listed under 
	#ENTRY in the ontology, specific to the chosen topic
	print("Building MeSH ID dictionary and topic-based term list.")
	mo_ids, mo_cats, mesh_term_list = build_mesh_dict(mo_filename) 
			
	print("Loaded %s MeSH terms and %s topic-relevant terms + variants." % \
			(len(mo_ids), len(mesh_term_list)))
	
	print("Building MeSH ID to ICD-10 dictionary.")
	#Build the MeSH to ICD-10 dictionary
	do_ids, do_xrefs_icd10, do_xrefs_terms = \
		build_mesh_to_icd10_dict(do_filename)
	
	#Load the sentence labels and vocabulary here.
	#Most of the vocabulary is inherited from MeSH terms.
	#Clean terms to produce stems
	print("Loading sentence classification labels and terms.")
	global sentence_labels
	with open(sentence_label_filename) as sentence_label_file:
		for line in sentence_label_file:
			splitline = (line.rstrip()).split(",")
			label = splitline[0]
			terms = splitline[1].split("|")
			clean_terms = []
			for term in terms:
				clean_terms.append(clean(term))
			sentence_labels[label] = clean_terms
	
	#Most sentence label terms are populated from MeSH terms
	#using the MeSH tree structure and its categories
	for cat in ["D03","D04","D25","D26","D27"]:
		for term in mo_cats[cat]:
			sentence_labels["DRUG"].append(term)
	for cat in ["C23"]:
		for term in mo_cats[cat]:
			sentence_labels["SYMP"].append(term)
		sentence_labels["SYMP"].remove("death") #Death is not a symptom.
	for cat in ["E01"]:
		for term in mo_cats[cat]:
			sentence_labels["PROC"].append(term)
	for cat in ["E02","E04"]:
		for term in mo_cats[cat]:
			sentence_labels["TREA"].append(term)
	for cat in ["F01","F03"]:
		for term in mo_cats[cat]:
			sentence_labels["LIFE"].append(term)
	for cat in ["M01"]:
		for term in mo_cats[cat]:
			sentence_labels["DEMO"].append(term)
	
	#Clean up the sentence labels a bit and stem
	
	for label in sentence_labels:
		for term in sentence_labels[label]:
			sentence_labels[label].remove(term)
			clean_term = clean(term)
			sentence_labels[label].append(clean_term)
			if len(clean_term) < 3:
				sentence_labels[label].remove(clean_term)
		sentence_labels[label] = set(sentence_labels[label])
			
	for label in sentence_labels:
		label_term_count = label_term_count + len(sentence_labels[label])
	print("Sentence label dictionary includes %s terms." % \
			label_term_count)
	
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
		print("Loading %s..." % medline_file)
		with open(medline_file) as this_medline_file:
			
			filereccount = 0

			fileindex = 0
			
			#Count entries in file first so we know when to stop
			#Could just get length of records but that's too slow
			#So just count entries by PMID
			filereccount = 0
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
			
			#for word in mesh_term_list:
			#	heart_word_list.append(word)
			
			have_records = True
			
			while record_count < record_count_cutoff \
					and record_count < filereccount \
					and fileindex <= filereccount:
				
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
					
					for word in heart_word_list:
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
						 
						 Also save the sentences in each 
						 abstract to a different file set (train + test)
						 for sentence classification purposes.
						 This is handled by the same function as
						 that used to save tag classifier training
						 text (save_train_or_test_text).
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
								this_sent_count = \
								save_train_or_test_text(these_clean_mesh_terms,
													record['TI'],
													record['AB'],
													record['PMID'],
													"test")
							else:
								this_sent_count = \
								save_train_or_test_text(these_clean_mesh_terms, 
													record['TI'],
													record['AB'],
													record['PMID'],
													"train")
							sent_count = sent_count + this_sent_count
							ti = ti+1
							
					#else:
					#	print("\nNOT MATCHING: %s" % record['PMID'])
							
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
		#broken right now
		#new_abstracts = {}
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
	
	print("Saved %s sentences for the sentence classifier." % sent_count)
	
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
	
	#Now labeled sentences are used to build a classifier
	#so newly provided sentences can be labelled.
	#Unlike MeSH terms, sentence labels are all new.
	print("\n\nStarting to build sentence label classifier.")
	sent_classifier, slb = sent_classification(testing)
	
	if not os.path.isdir("output"):
		#print("Setting up sentence classifier %sing directory." % cat)
		os.mkdir("output")
	os.chdir("output")
	
	if match_record_count < record_count_cutoff:
		filelabel = match_record_count
	else:
		filelabel = record_count_cutoff
		
	if len(medline_file_list) == 1:
		outfilename = (medline_file_list[0])[6:-4] + "_%s_relabeled.txt" \
						% filelabel
	else:
		outfilename = "medline_entries_%s_relabeled.txt" \
						% filelabel
	
	if len(medline_file_list) == 1:
		sent_outfilename = (medline_file_list[0])[6:-4] + "_%s_sentences.txt" \
						% filelabel
	else:
		sent_outfilename = "medline_entries_%s_sentences.txt" \
						% filelabel
						
	if len(medline_file_list) == 1:
		topic_outfilename = (medline_file_list[0])[6:-4] + "_%s_topics.txt" \
						% filelabel
	else:
		topic_outfilename = "medline_entries_%s_topics.txt" \
						% filelabel
	
	print("\nWriting matching, newly annotated records to file.")
	
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
					for sentence in sentence_list:
						clean_string = clean(sentence)
						clean_array = np.array([clean_string])
						predicted = sent_classifier.predict(clean_array)
						labels = slb.inverse_transform(predicted)
						flatlabels = [label for labeltuple in labels for label in labeltuple]
						cleanlabels = []
						
						#Ensure that NONE is used properly
						if len(flatlabels) > 1:
							for label in flatlabels:
								if label != "NONE":
									cleanlabels.append(label)
						else:
							cleanlabels = flatlabels
							
						labeled_abstract.append([sentence, cleanlabels])
				except UnicodeDecodeError:
					labeled_abstract = "UNLABELED ABSTRACT: %s" % abstract
				
				labeled_record['labstract'] = labeled_abstract
			
		labeled_ann_records.append(labeled_record)
					
	#Writing labeled abstracts to files
	print("\nWriting labeled sentences from all matching records.")
	
	with open(sent_outfilename, 'w') as outfile:
		for record in labeled_ann_records:
			outstring = "%s\n%s\n%s\n\n" % \
						(record['TI'], record['PMID'], record['labstract'])
			outfile.write(outstring)
			
	#Writing topics to files
	#For now, this is just a filter/rearrangement of the labeled sentences
	print("\nWriting topics for all matching records.")
	
	with open(topic_outfilename, 'w') as outfile:
		for record in labeled_ann_records:
			topics = []
			for labeled_sentence in record['labstract']:
				if labeled_sentence[1] != ['NONE']:
					for label in labeled_sentence[1]:
						topic = "%s: %s" % (label, labeled_sentence[0])
						topics.append(topic)
			outstring = "%s\n%s\n%s\n\n" % \
						(record['TI'], record['PMID'], '\n'.join(topics))
			outfile.write(outstring)
	
	#Now provide some summary statistics	
	high_match_terms = sorted(matched_mesh_terms.items(), key=operator.itemgetter(1),
								reverse=True)[0:15]
	
	high_terms_in_matched = sorted(all_terms_in_matched.items(), key=operator.itemgetter(1),
								reverse=True)[0:15]
	
	high_match_journals = sorted(matched_journals.items(), key=operator.itemgetter(1),
								reverse=True)[0:15]
								
	high_match_years = sorted(matched_years.items(), key=operator.itemgetter(1),
								reverse=True)[0:10]
								
	high_rn_codes = sorted(rn_codes.items(), key=operator.itemgetter(1),
								reverse=True)[0:10]
	
	if record_count > 0:
		print("\nTotal records: %s" % record_count)
		print("\nRecords with matched term in the title or MeSH terms: %s" % 
				match_record_count)
		print("\nNumber of records with abstracts: %s" % abstract_count)
		print("\nNumber of records with material codes: %s" % rn_counts)
		print("\nTotal unique material codes used: %s" % len(rn_codes))
		print("\nMost frequently matched MeSH terms:")
		for match_count in high_match_terms:
			print(match_count)
		print("\nMost common MeSH terms in matched records:")
		for match_count in high_terms_in_matched:
			print(match_count)
		print("\nMost records are from these journals:")
		for match_count in high_match_journals:
			print(match_count)
		print("\nMost records were published in these years:")
		for match_count in high_match_years:
			print(match_count)
		print("\nMost frequently used material codes:")
		for code in high_rn_codes:
			print(code)
	else:
		sys.exit("Found no matching references.")
	
	print("\nDone - see the following files in the output folder:\n"
			"%s for list of matching records,\n"
			"%s for labeled abstract sentences, and\n"
			"%s for topics within each entry." %
			(outfilename, sent_outfilename, topic_outfilename))
	
if __name__ == "__main__":
	sys.exit(main())
