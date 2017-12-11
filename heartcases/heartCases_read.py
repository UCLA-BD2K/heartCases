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

The topic_headings.txt file contains sets of codes to include as MeSH 
terms used to search documents with. Each set (actually a dict) is 
topic-specific. These topics are also used to train the label expansion 
classifier. For example, a heading of C14 and codes of 240 and 280 will 
include all terms under the heading C14 (Cardiovascular Diseases) but
only under its subheadings 240 and 260. For more specific subheadings,
these codes must be more specific as well: e.g. use 907.253.535 to only
use MeSH terms related to statins, use the heading D27 and the code
505.519.186.071.202.370 (corresponding to the full MeSH tree number of
D27.505.519.186.071.202.370).

To specify individual search terms rather than topics, see the
--terms argument.

Here, the all_cardiovascular term set is the default topic.
See topic_headings.txt for others.
'''
__author__= "Harry Caufield"
__email__ = "j.harry.caufield@gmail.com"

import pip #Just using it to check on installed packages
installed = pip.get_installed_distributions()
install_list = [item.project_name for item in installed]
need_install = []
for name in ["bokeh", "nltk", "sklearn","tqdm"]:
	if name not in install_list:
		need_install.append(name)
if len(need_install) > 0:
	sys.exit("Please install these Python packages first:\n%s" 
				% "\n".join(need_install))

import argparse, glob, operator, os, random, re, string, sys, time
import urllib, urllib2
import warnings

from itertools import tee, izip
from operator import itemgetter

from bokeh.layouts import column
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, WhitespaceTokenizer

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics, preprocessing
from sklearn.externals import joblib

from tqdm import *

from heartCases_help import find_citation_counts

#Constants and Options
						
abst_len_cutoff = 200
	#Abstracts less than this number of characters are considered
	#too short to use for classifier training or re-labeling.
	
train_on_target = True
'''
If True, trains the classifier to label only the search terms
included in mesh_topic_tree below.
If False, trains classifier on *all* terms used in training data.
This usually means too many different labels will be included
in the classification task and it will be very inefficient.
'''

topic_trees = {}
with open("topic_headings.txt") as headings_file:
	for line in headings_file:
		splitline = (line.rstrip()).split("\t")
		topic_name = splitline[0]
		topic_heading = splitline[1]
		topic_subheadings = splitline[2:]
		if topic_name not in topic_trees:
			topic_trees[topic_name] = {topic_heading:topic_subheadings}
		else:
			topic_trees[topic_name].update({topic_heading:topic_subheadings})
			
named_entities = {}
	#Dict of named entities with entity types as keys and sets
	#of terms as values.

#Classes
class Record(dict):
	'''
	Just establishes that the record is a dict
	'''

#Functions

def find_more_record_text(rec_ids):
	'''
	Retrieves abstract text (or more, if possible) for records lacking 
	abstracts. Takes dict with pmids as keys and pmc_ids as values as 
	input. Returns dict pmids as keys and abstract texts as values.
	
	Retrieves abstracts from PubMed Central if available.
	Doesn't search anything else yet.
	
	This list is processed all at once in order to control
	the number of requests made through NCBI resources.
	As this may still involve >500 IDs then NCBI requires
	call to the History server first as per this link:
	https://www.ncbi.nlm.nih.gov/books/NBK25499/
	Note this needs to be a POST request, not a GET
	'''
	
	pmc_ids_to_search = []
	newtext_dict = {} #PubMed IDs are keys, abstracts are values
	
	outfilepath = "Additional_PMC_records.txt"
	baseURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
	epost = "epost.fcgi"
	efetch = "efetch.fcgi?db=pmc"
	efetch_options = "&usehistory=y&retmode=text&rettype=medline"
	
	outfiledir = "output"
	outfilename = "Additional_PMC_entries.txt"
	
	if not os.path.isdir(outfiledir):
		os.mkdir(outfiledir)
	os.chdir(outfiledir)
	
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
				
				out_file = open(outfilename, "a")
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
			
			records = medline_parse(open(outfilename))
			
			for record in records:
				if 'AB' in record.keys() and 'PMID' in record.keys():
					pmid = record['PMID']
					ab = record['AB']
					newtext_dict[pmid] = ab 
	
			print("\nRetrieved %s new abstracts from PubMed Central." \
					% len(newtext_dict))
			
			os.chdir("..")
			
			return newtext_dict
		
		except urllib2.HTTPError as e:
			print("Couldn't complete PubMed Central search: %s" % e)
			os.chdir("..")
	
	else:
		print("No PubMed Central IDs found and/or all records with" \
				" PMC IDs already have abstracts.")
		os.chdir("..")
			
def get_data_files(name):
	'''
	Retrieves one of the following:
	the Disease Ontology database (do),
	the 2018 MeSH term file from NLM (mo),
	or the 2017 SPECIALIST Lexicon from NLM (sl).
	The last of these requires decompression and returns a directory.
	The others return a filename.
	'''
	
	data_locations = {"do": ("http://ontologies.berkeleybop.org/","doid.obo"),
					"mo": ("ftp://nlmpubs.nlm.nih.gov/online/mesh/MESH_FILES/asciimesh/","d2018.bin"),
					"sl": ("https://lexsrv3.nlm.nih.gov/LexSysGroup/Projects/lexicon/2017/release/LEX/", "LEXICON")}
	
	baseURL, filename = data_locations[name]
	filepath = baseURL + filename
	outfilepath = filename
		
	print("Downloading from %s" % filepath)
	try:
		response = urllib2.urlopen(filepath)
		out_file = open(os.path.basename(filename), "w+b")
		chunk = 1048576
		pbar = tqdm(unit="Mb")
		while 1:
			data = (response.read(chunk)) #Read one Mb at a time
			out_file.write(data)
			if not data:
				pbar.close()
				print("\n%s file download complete." % filename)
				out_file.close()
				break
			pbar.update(1)
	except urllib2.URLError as e:
		sys.exit("Encountered an error while downloading %s: %s" % (filename, e))
	
	return filename
	
def parse_disease_ontology(do_filename):
	'''
	Build the MeSH ID to ICD-10 dictionary
	the relevant IDs are xrefs in no particular order
	Also, terms differ in the xrefs provided (e.g., a MeSH but no ICD-10)
	So, re-assemble the entries first and remove those without both refs
	'''
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
	'''
	Loads the terms in the SPECIALIST lexicon.
	Does not process all fields.
	Not currently used.
	'''
	
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

def build_mesh_dict(mo_filename, mesh_topic_tree):
	'''
	Sets up the dict of MeSH terms, specific to the chosen topic.
	The subset of terms to select (the topic) is defined by global 
	variable mesh_topic_tree above.
	'''
	
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
				add_synonyms = False
				#Location in the MeSH tree. May have multiple locations
				code = (line.split("="))[1].strip()
				codetree = code.split(".")
				tree_cat = codetree[0]
				
				if codetree[0] in mesh_topic_tree:

					if len(codetree) > 1: 
						#This is any set of secondary subheadings
						#e.g if tree is C14.907.253.061
						#this may match 907 or 907.253
						if codetree[1] in mesh_topic_tree[codetree[0]]:
							add_synonyms = True
						elif ".".join(codetree[1:]) in mesh_topic_tree[codetree[0]]:
							#Note that this requires an exact match
							add_synonyms = True
						
				if add_synonyms:
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
	'''
	Given a file containing a list of PMIDs, one per line.
	Creates a file containing one MEDLINE record for each PMID.
	Returns name of this file.
	'''
	
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
	pbar = tqdm(unit="Mb")
	while 1:
		data = (response.read(chunk)) #Read one Mb at a time
		out_file.write(data)
		if not data:
			pbar.close()
			print("\nRecords retrieved - see %s" % outfilepath)
			out_file.close()
			break
		pbar.update(1)
		
	return outfilepath
	
def get_mesh_terms(terms_list):
	'''
	Given a file containing a list of MeSH terms, one per line.
	Returns a list of terms. Pretty simple.
	'''
	
	terms = []
	
	with open(terms_list) as infile:
		for line in infile:
			terms.append((line.rstrip()).lower())
	
	return terms
	
def medline_parse(medline):
	'''
	Parses a MEDLINE file by heading
	Returns fields (out of order, unfortunately)
	Also squishes records if they have the same key value.
	e.g. all "AD" values end up in one long string
	
	Yields a dict object from a generator object.
	
	'''
	
	strkeys = ("AB","AD","CA","CY","DA","DCOM","DEP","DP","EA","EDAT",
				"ID","IP","IS","JC","JID","JT","LID","LR","MHDA","NI",
				"OAB","OWN","PG","PL","PMC","PMID","PST","PUBM","RF",
				"SB","SO","STAT","TA","TI","TT","VI","YR")
				#These are NOT all the keys in a MEDLINE file,
				#these are just the keys to treat as strings rather than 
				#as lists.
			 
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
	for training or testing a classifier for MeSH term expansion.
	
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

def label_this_text(text, verbose=False):
	'''
	The NER labeler.
	
	Takes a string (usually a sentence or abstract) as input.
	Labels named entities within the string using the term dictionary.
	Returns a list of lists of the form:
	[label_name, start, end, text]
	where
	label_name is the name of the corresponding named entity
	start is the position of the character where the label begins
	end is the position of the character where the label ends
	text is the full text of the labeled entity.
	
	If no entities are found then the list is [["NONE",0,0,"NA"]].

	Labels terms up to n words long, without splitting terms
	across punctuation.
	
	Stopwords are not included unless they are within an n-gram,
	e.g. "inflammation of the esophagus" is OK but "inflammation of"
	is not.
	'''
	
	stopword_set = set(stopwords.words('english'))
	
	spans = []
	tokens = []
	split_text = [] 
	labels = []
	
	string_word_count = len(text.split())
	
	t0 = time.time()
	
	'''
	Populate split_text with lists, where each list is a n-gram
	of size between 1 and n, inclusive,
	and the start and end indices of the n-gram,
	 of the form [start, n_gram, end].
	Note that start is the first character index
	and end is the index after the final character.
	'''
	
	span_gen = WhitespaceTokenizer().span_tokenize(text)
	single_spans = [span for span in span_gen]
	single_words = WhitespaceTokenizer().tokenize(text)
	spans = spans + single_spans
	tokens = tokens + single_words
	
	i = 0
	for span in spans: #Add span indices and contents
		start = span[0]
		end = span[1]
		word = tokens[i]
		i = i +1
		split_text.append([start,word,end])
	
	split_text_multiword = []
	for n in range(1, 6): #Add multiword tokens, up to 6 words
		i = 0
		for token_and_index in split_text:
			token_ok = True
			try:
				start = split_text[i][0]
				end = split_text[i+n][2]
				token = ""
				for text in split_text[i:i+n+1]:
					word = text[1]
					if word.lower() in stopword_set: #Don't extend spans with stopwords alone
						token_ok = False
						break
					token = token + " " + text[1]
				if token_ok:
					token = token.strip()
					split_text_multiword.append([start,token,end])
				i = i +1
			except IndexError:
				break
	
	split_text = split_text + split_text_multiword
				
	#Add labels from named_entities set
	#Check for overlap
	for ne_type in named_entities:

		for token_and_index in split_text:
			add_this_label = False
			
			token = token_and_index[1]
			clean_token = clean(token)
			
			if clean_token in named_entities[ne_type]:
				add_this_label = True
				start = token_and_index[0]
				end = token_and_index[2]
				label_size = end - start
				for label in labels:
					if start == label[1] or end == label[2]:
						other_label_size = label[2] - label[1]
						if label_size <= other_label_size:
							add_this_label = False
						break
				if add_this_label:
					labels.append([ne_type, start, end, token])
					
	
	#Labels are sorted by starting character
	sortedlabels = sorted(labels, key=itemgetter(1))
	labels = sortedlabels
	
	t1 = time.time()
	totaltime = t1 - t0
	
	if verbose:
		print("Labeled text of %s words and %s tokens with %s labels in %.2f sec."
					% (string_word_count, len(split_text), len(labels), 
						totaltime))
	
	return labels
		
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

def clean(text, no_stem=False):
		'''
		Pre-processing for a string to ensure it lacks stopwords, etc.
		Also removes punctuation.
		Uses NLTK Snowball stemmmer.
		Returns the input string as a processed raw string.
		
		If no_stem is True, then the string is not stemmed.
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
			if not no_stem:
				word = stemmer.stem(word)
			if word.lower() not in stopword_set: #No stopwords
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
	Uses scikit-learn's DecisionTree implementation.
	
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
			if len(train_file_list) > 0:
				print("Loaded %s labeled abstracts." % len(train_file_list))
			else:
				sys.exit("No labeled abstracts found for training. Exiting...")
			
		if testing:
			print("Loading labeled abstracts for testing.") 
			test_file_list = glob.glob('training_test_text/*.txt')
		
			for test_file in test_file_list:
				with open(test_file) as tfile:
					labeled_text = parse_training_text(tfile)
					all_test_text = all_test_text + labeled_text
			if len(test_file_list) > 0:
				print("Loaded %s labeled abstracts." % len(test_file_list))
			else:
				sys.exit("No labeled abstracts found for testing. Exiting...")
		
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
	#and we need to convert labels to binary values
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
		
		print("Learning MeSH term dictionary...")
		#Get all MeSH terms in use in the training set
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
			
		if X_train_size < 5:
			sys.exit("Not enough abstracts to train classifier. Exiting...")
			
		t1 = time.time()
		'''
		This is a scikit-learn pipeline of:
		A vectorizer to extract counts of ngrams, up to 3,
		a tf-idf transformer, and
		a DecisionTreeClassifier, used once per label (OneVsRest) to 
		perform multilabel classification.
		'''
		classifier = Pipeline([
					('vectorizer', HashingVectorizer(analyzer='word',
									ngram_range=(1,3),
									stop_words = 'english',
									norm='l2')),
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
		
		X_test_size = len(X_test_pre)
		if X_test_size < 5:
			sys.exit("Not enough abstracts to test classifier. Exiting...")
		
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

def plot_those_counts(counts, all_matches, outfilename, ptitle):
	'''
	Given a dict (with names as values) - all_matches here - 
	of dicts of categories and counts,
	produces a simple bar plot of the counts for each.
	Also takes simple counts to be provided as numbers
	rather than plots (in the counts variable here).
	Produces plots using Bokeh - produces .html with outfilename
	as title and opens browser.
	Title variable provides plot title.
	'''
	all_plots = []
	
	#Plot simple counts first
	height = (len(counts)*100) + 200
	textplot = figure(plot_width=700, plot_height=height, 
					title=ptitle)
	textplot.axis.visible = False
	
	i = 100
	for count_name in counts:
		this_count = counts[count_name]
		
		plot_string = "%s: %s" % (count_name, str(this_count))
		
		count_label = Label(x=25, y=i, text=plot_string,
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
		plot.hbar(y = catvalues, left=0, right=counts, height=0.66, color=randcol)
		
		plot.text(0, catvalues, text=[i for i in cats], 
					text_font_size=text_size, text_baseline="middle")
		
		all_plots.append(plot)
	
	output_file(outfilename, mode="inline", title="heartCases")
	
	#For gridding
	layout = column(all_plots)
	show(layout)
	#show(plot)	

def populate_named_entities(named_entities, mo_cats, do_xrefs_terms):
	#Sets up the named entity vocabulary.
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
		if cat[:1] == "A" and cat not in ["A18","A19","A20","A21",]:
			for term in mo_cats[cat]:
				named_entities["body_part"].append(term)
		if cat[:1] == "C":
			if cat not in ["C23","C26"]:
				for term in mo_cats[cat]:
					named_entities["disease"].append(term)
			elif cat == "C23":
				for term in mo_cats[cat]:
					named_entities["symptom"].append(term)
			elif cat == "C26":
				for term in mo_cats[cat]:
					named_entities["wound"].append(term)
		if cat in ["D03","D04","D25","D26","D27"]:
			for term in mo_cats[cat]:
				named_entities["drug"].append(term)
		if cat in ["E01","E04","E05"]:
			for term in mo_cats[cat]:
				named_entities["technique"].append(term)
		if cat in ["E07"]:
			for term in mo_cats[cat]:
				named_entities["equipment"].append(term)
		if cat in ["G09"]:
			for term in mo_cats[cat]:
				named_entities["cardio_phenomenon"].append(term)
		if cat in ["I03"]:
			for term in mo_cats[cat]:
				named_entities["activity"].append(term)
		if cat in ["J02"]:
			for term in mo_cats[cat]:
				named_entities["food"].append(term)
		if cat in ["M01"]:
			for term in mo_cats[cat]:
				named_entities["person_detail"].append(term)
		
	#Add disease names from the Disease Ontology
	for code in do_xrefs_terms:
		for term in code:
			named_entities["disease"].append(term)

	#Convert list items in named_entities to sets for efficiency.
	for ne_type in named_entities:
		named_entities[ne_type] = set(named_entities[ne_type])
	
	#Then clean up terms and stem them with clean()
	#Ignore some terms we know are not useful for labeling
	#or are frequently mis-labeled
	new_named_entities = {}
	
	stop_terms = ["extremity","report"]
	
	for ne_type in named_entities:
		new_terms = set()
		for term in named_entities[ne_type]:
			if term not in stop_terms:
				clean_term = clean(term)
				if len(clean_term) > 2:
					new_terms.add(clean_term)
		new_named_entities[ne_type] = new_terms
		
	named_entities = new_named_entities
	
	if not os.path.isdir("output"):
		os.mkdir("output")
	os.chdir("output")
	
	ne_dump_filename = "NE_dump.tsv"
	with open(ne_dump_filename, 'w') as outfile:
		print("Dumping named entities to file: %s" % ne_dump_filename) 
		for ne_type in named_entities:
			for term in named_entities[ne_type]:
				outfile.write("%s\t%s\n" % (ne_type, term))
				
	os.chdir("..")
		
	return named_entities

def parse_args():
	#Parse command line arguments for heartCases_read
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--citation_counts', help="if TRUE, retrieve "
						"citation counts for matching records")
	parser.add_argument('--citation_range', nargs = 2,
						type=int, help="Takes two integers. All "
						"matched records will have at least the first "
						"specified number of citations from PubMed Central "
						"entries and no more than the second number."
						"Requires citation_counts option to be TRUE.")
	parser.add_argument('--inputfile', help="name of a text file containing "
						"MEDLINE records")
	parser.add_argument('--mesh_expand', help="if FALSE, do not perform "
						"MeSH term expansion. Can save time.")
	parser.add_argument('--ner_label', help="if FALSE, do not perform "
						"NER labeling on any documents. Can save time.")
	parser.add_argument('--pmids', help="name of a text file containing "
						"a list of PubMed IDs, one per line, to retrieve "
						"MEDLINE records for")
	parser.add_argument('--recordlimit', help="The maximum number of "
						"records to search.")
	parser.add_argument('--search_topic', help="Select a pre-curated set of "
						"MeSH terms to search with and to use for term "
						"expansion. See topic_headings.txt for topic "
						"names and corresponding headings.")
	parser.add_argument('--terms', help="name of a text file containing "
						"a list of MeSH terms, one per line, to search for. "
						"Will also search titles for terms, unless "
						"--search_doc_titles is set to FALSE.")
	parser.add_argument('--testing', help="if FALSE, do not test classifiers")
	parser.add_argument('--verbose', help="if TRUE, provide verbose output")
	parser.add_argument('--first_match', help="if TRUE, filter input "
						"based on first match but do not count all "
						"matches. Can save time.")
	parser.add_argument('--search_doc_titles', help="if FALSE, do not "
						"search document titles.")
	parser.add_argument('--random_records', help="if TRUE, choose "
						"training records at random until hitting "
						"value specified by --recordlimit argument.")
	
	try:
		args = parser.parse_args()
	except:
		sys.exit()
	
	return args

def setup_labeledfiledir(named_entities):
	#Prepares configuration documents for BRAT format NER label files.
	ann_conf_filename = "annotation.conf"
	vis_conf_filename = "visual.conf"
	
	#Set up entity names
	if not os.path.isfile(ann_conf_filename):
		with open(ann_conf_filename, 'w') as outfile:
			outfile.write("[entities]\n\n")
			for ne in named_entities:
				outfile.write("%s\n" % ne)
			outfile.write("\n")
			outfile.write("[attributes]\n\n") #These are just placeholders for now
			outfile.write("[relations]\n\n")
			outfile.write("[events]\n\n")
	
	#Set up visual properties for labels
	#Assigns random colors for now just to distinguish	
	ne_label_colors = {}
	for ne in named_entities:
		bgColor = ('#%06X' % random.randint(0,256**3-1))
		ne_label_colors[ne] = {"fgColor":"black", 
									"bgColor":bgColor, 
									"borderColor":"darken"}
							
	if not os.path.isfile(vis_conf_filename):
		with open(vis_conf_filename, 'w') as outfile:
			outfile.write("[labels]\n\n")
			for ne in named_entities:
				outfile.write("%s\n" % ne)
			outfile.write("\n\n[drawing]\n\n")
			for ne in ne_label_colors:
				outfile.write("%s fgColor:%s, bgColor:%s, borderColor:%s\n" 
								% (ne, 
									ne_label_colors[ne]["fgColor"],
									ne_label_colors[ne]["bgColor"],
									ne_label_colors[ne]["borderColor"]))

#Main
def main():
	
	warnings.simplefilter("ignore", UnicodeWarning)
	
	record_count = 0 #Total number of records searched, across all files
	match_record_count = 0 #Total number of records matching search terms
							#For now that is keywords in title or MeSH
	abstract_count = 0 #Number of abstracts among the matching records
	rn_counts = 0 #Number of abstracts containing RN material codes
	rn_codes = {} #Keys are RN codes in use, values are counts
	matched_mesh_terms = {} #Keys are terms, values are counts
	matched_journals = {} #Keys are journal titles, values are counts
	matched_years = {} #Keys are years, values are counts
	citation_counts = {} #Keys are citation counts (where 0 is no
							#citations, 1 is 1 citation, etc)
							#and values are counts of that citation count
	all_terms_in_matched = {} #Counts of all MeSH terms in matched records
								#Keys are terms, values are counts
	all_pmids = []		#A list of all PMIDs in matched records
						#Used to look up summary details from Pubmed
	fetch_rec_ids = {} #A dict of record IDs for those records
						#to be searched for additional text retrieval
						#PMIDs are keys, PMC IDs are values
						#or, if no PMC ID available, PMC ID is "NA"
	new_abstract_count = 0 #Number of records with abstracts not directly
							#available through PubMed
	ne_count = 0 #Count of all terms in named_entities
	
	citation_range = [0,9999] #Default citation range
	
	print("*** heartCases - read module ***")
	
	#Set up parser
	args = parse_args()
	
	#Create input folder if it doesn't exist
	if not os.path.isdir("input"):
		os.mkdir("input")
	
	#Parse some arguments provided
	verbose = False
	if args.verbose:
		if args.verbose == "TRUE":
			verbose = True
			
	get_citation_counts = False
	use_citation_range = False
	if args.citation_counts:
		if args.citation_counts == "TRUE":
			get_citation_counts = True
			if args.citation_range:
				if args.citation_range != citation_range:
					use_citation_range = True
					citation_range = args.citation_range
		
	mesh_expand = True
	if args.mesh_expand:
		if args.mesh_expand == "FALSE":
			mesh_expand = False
			
	if args.search_topic:
		try:
			mesh_topic_tree = topic_trees[args.search_topic]
		except KeyError:
			print("Couldn't find requested search topic. Using default.")
			mesh_topic_tree = topic_trees["cardiovascular"]
	else:
		mesh_topic_tree = topic_trees["cardiovascular"]

	#Argument tells us if we should not test the classifier
	#This saves some time.
	testing = True
	if args.testing:
		if args.testing == "FALSE":
			testing = False
			
	ner_label = True
	if args.ner_label:
		if args.ner_label == "FALSE":
			ner_label = False
			
	#Argument tells us if we should count all MeSH term matches
	#in title and terms or just the first one.
	#This saves time if there are many documents or search terms.
	first_match_only = False
	if args.first_match:
		if args.first_match == "TRUE":
			first_match_only = True
			
	search_doc_titles = True
	if args.search_doc_titles:
		if args.search_doc_titles == "FALSE":
			search_doc_titles = False
	
	#If True, choose training records at random until hitting
	#record_count_cutoff value
	random_record_list = False
	if args.random_records:
		if args.random_records == "TRUE":
			random_record_list = True
	
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
	mesh_ofile_list = glob.glob('d2018.*')
	if len(mesh_ofile_list) >1:
		print("Found multiple possible MeSH term files. "
				"Using the preferred one.")
		mo_filename = "d2018.bin"
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
		
	mo_ids, mo_cats, mesh_term_list = build_mesh_dict(mo_filename, mesh_topic_tree) 
	
	unique_term_count = len(set(mo_ids.values()))
	synonym_count = len(mo_ids) - unique_term_count
	print("Full MeSH ontology includes:\n"
			"%s unique MeSH terms and %s synonyms \n"
			"across %s primary subheadings." % \
			(unique_term_count, synonym_count, len(mo_cats)))
	
	#Check if custom MeSH search term list was provided.
	#If so, just search for these
	#Otherwise, just use CVD-specific MeSH terms produced above
	#to filter for on-topic documents.
	if args.terms:
		terms_file = str(args.terms)
		print("Using MeSH terms in the following file to search records: "
				"%s" % terms_file)
		custom_mesh_terms = get_mesh_terms(terms_file)
		have_custom_terms = True
		print("File contains %s terms." % len(custom_mesh_terms))
	else:
		print("Searching documents using all topic-related terms.")
		print("List includes %s topic-relevant terms + synonyms." % \
			(len(mesh_term_list)))
		have_custom_terms = False
		
	print("Building MeSH ID to ICD-10 dictionary using Disease Ontology.")
	#Build the MeSH to ICD-10 dictionary
	do_ids, do_xrefs_icd10, do_xrefs_terms = \
		parse_disease_ontology(do_filename)
		
	#Load the SPECIALIST lexicon if needed
	#Not currently in use, however.
	#if ner_label:
	#	print("Loading SPECIALIST lexicon...")
	#	slexicon = load_slexicon(sl_filename)
	#	print("Loaded %s lexicon entries." % len(slexicon))
	
	#Check for and download NLTK corpus if needed
	print("Checking for NLTK resources...")
	nltk.download('stopwords')
	
	#Load the named entities here if needed
	#Most of the vocabulary is inherited from MeSH terms.
	#Clean terms to produce stems.
	if ner_label:
		
		print("Loading named entity dictionary...")
		
		global named_entities
		
		named_entities = populate_named_entities(named_entities, mo_cats, do_xrefs_terms)
		
		for ne_type in named_entities:
			ne_count = ne_count + len(named_entities[ne_type])
	
		print("Named entity dictionary includes %s terms across these entity types:" % \
				ne_count)
		for ne_type in named_entities:
			print("%s %s" % (str(ne_type).ljust(30, ' '),
							len(named_entities[ne_type])))
	
	#Check if a recordlimit has been set.
	if args.recordlimit:
		record_count_cutoff = int(args.recordlimit)
	else:
		record_count_cutoff = 9999999
	
	if get_citation_counts:
		print("Will retrieve citation counts prior to searching records.")
	
	if use_citation_range:
		print("Will return matches only for documents with citation "
				"counts from PubMed Central entries of at least "
				"%s and no more than %s." % (citation_range[0], 
				citation_range[1]))
	
	ti = 0
	
	matching_orig_records = []
	
	filtered_count = 0 #If filters in use, count how many records
								#get filtered out
	
	all_citation_counts = {}
	
	all_raw_cite_counts = {}
	
	for medline_file in medline_file_list:
		print("\nLoading %s..." % medline_file)
		with open(medline_file) as this_medline_file:
			
			filereccount = 0 #Total number of records in this file

			fileindex = 0 #Index of records in this file (starting at 0)
								
			these_pmids = []
			
			#Count entries in file first so we know when to stop
			#Could just get length of records but that's too slow
			#So just count entries by PMID

			for line in this_medline_file:
				if line[:6] == "PMID- ":
					filereccount = filereccount +1
					splitline = (line.rstrip()).split("-")
					these_pmids.append(splitline[1].lstrip())
			print("\tFile contains %s records." % filereccount)
			
			#Retrieve citation counts from PubMed, now that we have 
			#PMIDs to work with
			#This may take a while.
			if get_citation_counts:
				print("\nFinding citation counts for this file.")
				citation_counts, raw_cite_counts, citation_count_filename = \
					find_citation_counts(these_pmids)
				if len(raw_cite_counts) == 0:
					print("WARNING: No citation counts found!")
				all_citation_counts.update(citation_counts) 
					#Cite counts by pub name and bin
				all_raw_cite_counts.update(raw_cite_counts) 
					#Cite counts by PMID
				print("Now loading file...")
			
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
				pbar = tqdm(total=filereccount)
			else:
				pbar = tqdm(total=record_count_cutoff)

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
					
					#Get the Pubmed ID
					pmid = record['PMID']
					all_pmids.append(pmid)
					
					#If this PMID doesn't pass the filter (if any filters)
					#just move ahead
					if use_citation_range:
						filter_record = False
						if pmid in all_raw_cite_counts:
							this_cite_count = int(raw_cite_counts[pmid])
							if citation_range[0] == citation_range[1]:
								#We're just checking for identity.
								if this_cite_count != citation_range[0]:
									filter_record = True
							else:
								#We're checking for counts in range.
								if not citation_range[0] <= this_cite_count <= citation_range[1]:
									filter_record = True
						if filter_record:		
							filtered_count = filtered_count +1
							fileindex = fileindex +1
							continue
					
					try:
						#Some records don't have titles. Not sure why.
						clean_title = clean(record['TI'])
						split_title = (clean_title).split()
					except KeyError:
						split_title = ["NA"]
					
					if have_custom_terms:
						search_term_list = custom_mesh_terms
					else:
						search_term_list = mesh_term_list
					
					if search_doc_titles:
						for word in search_term_list:
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
						for term in search_term_list:
							if term in these_clean_mesh_terms:
								found = 1
								if term not in matched_mesh_terms:
									matched_mesh_terms[term] = 1
								else:
									matched_mesh_terms[term] = matched_mesh_terms[term] +1
								if first_match_only:
									break
							if term in these_clean_other_terms:
								found = 1
								if first_match_only:
									break
					
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
						#If there's no abstract, store PMC ID for this 
						#so we may retrieve abstract from there, if
						#possible
						if 'AB' in record.keys() and len(record['AB']) > abst_len_cutoff:
							abstract_count = abstract_count +1
							have_abst = 1
						else: 
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
						Save 80 percent of matching abstracts to 
						training files in the folder "training_text" and
						20 percent of matching abstracts to testing
						files in the folder "training_test_text".
						 Each file in these folders is one set of 
						 training text from one abstract. Each line of 
						 each file is one set of MeSH terms separated 
						 by |. The terms are followed by a tab, 
						 the full abstract text, and the PMID.
						 
						 If train_on_target is True,
						 these files are only those containing matching
						 *MeSH terms* specifically and *only* those
						 terms will be used as labels.
						 
						 Otherwise, ALL MeSH terms will be used to
						 train the classifier. This gets very large
						 and unwieldy very quickly.
						 
						 This is all skipped if mesh_expand is False.
						'''
						if mesh_expand:
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
								
								if ti % 5 == 0:
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
					
					pbar.update(1)

					if record_count == record_count_cutoff:
						break
	
	pbar.close()
	
	print("Done loading input file.")
	
	have_new_abstracts = False
	
	#Retrieve additional abstracts from PMC for all PMC IDs
	#Note that only a subset of records will have PMC IDs
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
	if mesh_expand:
		if train_on_target:
			print("\nStarting to build term classifier with target "
					"terms only...")
		else:
			print("\nStarting to build term classifier for all terms "
					"used in the training abstracts...")
	
		abst_classifier, lb = mesh_classification(testing)
		#Also returns the label binarizer, lb
		#So labels can be reproduced
		
		#This is the point where we need to add the new MeSH terms
		#and *then* search for new matching ICD-10 codes
		#Denote newly-added terms with ^
	
	matching_ann_records = []
		
	if matching_orig_records > 0:
		print("\nAdding new terms and codes to records.")
	else:
		sys.exit("\nNo matching records found in the input.")
		
	if len(matching_orig_records) > 10000:
		print("This may take a while.")
	
	#Progbar setup
	if not verbose:
		pbar = tqdm(total=len(matching_orig_records))
		
	j = 0
		
	for record in matching_orig_records:
		'''
		Use classifier on abstract to get new MeSH terms if
		mesh_expand is True,
		append any new terms to record["MH"], and
		add new ICD-10 codes.
		Ensure new terms are denoted properly as above.	
		This step can be very slow - 
		mostly due to classifier predictions.
		
		Some records may be filtered out at this point as well.
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
		t0 = time.time()
		if 'AB' not in record.keys() and have_new_abstracts:
			if record['PMID'] in new_abstracts:
				record['AB'] = new_abstracts[record['PMID']]
		
		#Now use MeSH term classifier to predict all possible 
		#associated terms
		if mesh_expand:
			if 'AB' in record.keys() and len(record['AB']) > abst_len_cutoff:
				titlestring = record['TI']
				abstring = record['AB']
				clean_string = "%s %s" % (clean(titlestring), clean(abstring))
				clean_array = np.array([clean_string])
				predicted = abst_classifier.predict(clean_array)
				all_labels = lb.inverse_transform(predicted)
				have_more_terms = True
		
		t1 = time.time()
		
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
		
		t2 = time.time()
		classtime = t1 - t0
		totaltime = t2 - t0
		
		if verbose:
			print("Processed record %s (%s out of %s) in %.2f sec total " \
					"(%.2f for classifier)"
					% (record['PMID'], j, len(matching_orig_records), 
						totaltime, classtime))
		else:
			pbar.update(1)

	pbar.close()
	
	'''
	Output the matching entries, complete with new annotations
	Note that, unlike in original MEDLINE record files,
	this output does not always place long strings on new lines.
	'''
	
	if match_record_count < record_count_cutoff:
		filelabel = match_record_count
	else:
		filelabel = record_count_cutoff
	
	#If this is a topic-based search, add corresponding topic name
	if args.search_topic:
		filelabel = "%s_%s" % (filelabel, args.search_topic)
		
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
	
	if not os.path.isdir("output"):
		os.mkdir("output")
	os.chdir("output")
	
	print("\nWriting matching, newly annotated full records to file...")
	
	#Progbar setup
	pbar = tqdm(total=len(matching_ann_records))
	j = 0
			
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
			
			j = j+1
		
			pbar.update(1)

	pbar.close()
	
	#Labeling sentences within the matching records
	#using the label_this_text function
	#unless ner_label is False
	if ner_label:
		print("\nTagging entities within results...")
		
		#Progbar setup
		if not verbose:
			pbar = tqdm(total=len(matching_ann_records))
		
		j = 0
		
		labeled_ann_records = []
		for record in matching_ann_records:
			labeled_record = {}
			labeled_record['TI'] = "NO TITLE"
			labeled_record['labstract'] = {'text':"NO ABSTRACT",'labels':[["NONE"]]}
			for field in record:
				if field == 'TI':
					labeled_record['TI'] = record[field]
				if field == 'PMID':
					labeled_record['PMID'] = record[field]
				if field == 'AB':
					abstract = record[field]
					labeled_abstract = {'text':"",'labels':[]}
					labeled_abstract['text'] = abstract
					labeled_abstract['labels'] = label_this_text(abstract, 
																verbose)
					
					labeled_record['labstract'] = labeled_abstract
				
			labeled_ann_records.append(labeled_record)
			
			j = j+1
			
			if not verbose:
				pbar.update(1)
		
		if not verbose:
			pbar.close()
				
		#Writing abstracts with labeled entities to files
		#both as one file with all labels
		#and as one file per document, with labels in BRAT format
		
		print("\nWriting raw entities and text found in abstracts...")
		
		with open(raw_ne_outfilename, 'w') as outfile:
			for record in labeled_ann_records:
				
				outfile.write("%s\n" % record['TI'])
				outfile.write("%s\n" % record['PMID'])
				outfile.write(str(record['labstract']))
				outfile.write("\n\n")
		
		print("\nWriting text with NER labels in BRAT format...")
		
		labeled_filedir = "brat"
		
		if not os.path.isdir(labeled_filedir):
			os.mkdir(labeled_filedir)
		os.chdir(labeled_filedir)
		
		setup_labeledfiledir(named_entities)
		
		for record in labeled_ann_records:
			
			txt_outfilename = "%s.txt" % record['PMID']
			with open(txt_outfilename, 'w') as outfile:
				labeled_abstract = record['labstract']
				outfile.write(labeled_abstract['text'])
				
			ann_outfilename = "%s.ann" % record['PMID']
			with open(ann_outfilename, 'w') as outfile:
				i = 1
				for label in record['labstract']['labels']:
					label_name = label[0]
					if label_name != "NONE":
						start = label[1]
						end = label[2]
						text = label[3]
						outfile.write("T%s\t%s %s %s\t%s\n" % (i, label_name, start, end, text))
						i = i+1
					
		os.chdir("..")
	
	if record_count > 0: #We can provide output
		
		output_file_dict = {outfilename: "full matching records with MEDLINE headings",
						viz_outfilename: "plots of metadata for matching records"}
		
		if ner_label:
			output_file_dict[raw_ne_outfilename] = "raw named entities found in abstracts"
			output_file_dict[labeled_filedir] = "directory for labeled documents in BRAT format"
		
		#Save some of the metadata counts to files.
		mtermfilename = "matched_mesh_terms.txt"
		with open(mtermfilename, "w+") as mtermfile:
			for item in matched_mesh_terms:
				outline = "%s\t%s\n" % (item, matched_mesh_terms[item])
				mtermfile.write(outline)
			output_file_dict[mtermfilename] = "counts of matching MeSH terms from the search terms"
		
		atermfilename = "all_terms_in_matched.txt"	
		with open(atermfilename, "w+") as atermfile:
			for item in all_terms_in_matched:
				outline = "%s\t%s\n" % (item, all_terms_in_matched[item])
				atermfile.write(outline)
			output_file_dict[atermfilename] = "counts of all MeSH terms in the documents"
		
		yearfilename = "matched_years.txt"
		with open(yearfilename, "w+") as yearfile:
			for item in matched_years:
				outline = "%s\t%s\n" % (item, matched_years[item])
				yearfile.write(outline)
			output_file_dict[yearfilename] = "counts of all publication years among the documents"
		
		rnfilename = "rn_codes.txt"	
		with open(rnfilename, "w+") as rnfile:
			for item in rn_codes:
				outline = "%s\t%s\n" % (item, rn_codes[item])
				rnfile.write(outline)
			output_file_dict[rnfilename] = "counts of all material codes among the documents"
		
		#Plot first.
		#Then provide some summary statistics.
		counts = {"Total searched records": record_count,
					"Number of records with abstracts": abstract_count,
					"Number of records with material codes": rn_counts,
					"Total unique material codes used": len(rn_codes)}
		all_matches = {"Most frequently matched MeSH terms": matched_mesh_terms,
						"Most common MeSH terms in matched records": all_terms_in_matched,
						"Most records are from these journals": matched_journals, 
						"Most records were published in these years": matched_years,
						"Most frequently used material codes": rn_codes}
		
		if filtered_count > 0:
			counts["Matched records after term matching and filtering"] = \
					match_record_count
			counts["Records not included due to filtering"] = \
					filtered_count
		else:
			if search_doc_titles:
				counts["Records with matched term in the title or MeSH terms"] = \
						match_record_count	
			else:
				counts["Records with matched term among MeSH terms"] = \
						match_record_count
							
		plot_those_counts(counts, all_matches, viz_outfilename, "Summary Counts")
		
		all_matches_high = {}
		for entry in all_matches:
			all_matches_high[entry] = sorted(all_matches[entry].items(), key=operator.itemgetter(1),
									reverse=True)[0:15]
		
		for entry in counts:
			print("\n%s:\t%s" % (entry, counts[entry]))
		
		for entry in all_matches_high:
			print("\n%s:\n" % entry)
			for match_count in all_matches_high[entry]:
				print("%s\t\t\t\t%s" % (str(match_count[0]).ljust(40, ' '), match_count[1]))
		
		#Citation report plots go in their own file.
		if get_citation_counts:
			cite_counts = {"Total records in input": record_count,
							"Different journals in input": len(all_citation_counts)}
					  
			cite_viz_outfilename = "citations_by_journal_report.html"
			plot_those_counts(cite_counts, all_citation_counts, cite_viz_outfilename,
								"Citation counts by journal")
			output_file_dict[cite_viz_outfilename] = "plots of citation counts by journal"
		
		os.chdir("..")
			
	else:
		sys.exit("Found no matching references.")
	
	print("\nDone - see the following files in the output folder:\n")
	for item in output_file_dict:
		print("%s\t\t%s" % (item.ljust(40, ' '), output_file_dict[item]))
	
	if get_citation_counts:
		("\nSee %s for PMIDs and citation counts." % citation_count_filename)
	
if __name__ == "__main__":
	sys.exit(main())
