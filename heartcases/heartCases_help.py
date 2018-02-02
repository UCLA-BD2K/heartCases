#!/usr/bin/python
#heartCases_help.py
'''
heartCases is a medical language processing system for case reports 
involving cardiovascular disease (CVD).

This part of the system includes helper functions.

'''

import os, sys, urllib, urllib2
from tqdm import *

def build_mesh_to_icd10_dict(icd10_map_files):
	'''
	Build the MeSH ID to ICD-10 dictionary.
	This is built on the Disease Ontology ID maps at the moment -
	provided through the icd10_map_files value (a list).
	Terms without direct cross-references inherit their parental xrefs.
	
	A second map may be provided - this is a WIP.
	
	The relevant IDs are xrefs in no particular order.
	Also, terms differ in the xrefs provided (e.g., a MeSH but no ICD-10)
	so, re-assemble the entries first and remove those without both refs.
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
	
	do_filename = icd10_map_files[0]
	
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

def get_file_and_append(pmids, filename):
	#Given a list of PMIDs, download PubMed records using eutils
	#and history server. Append to the file if it already exists.
	
	baseURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
	epost = "epost.fcgi"
	esummary = "esummary.fcgi?db=pubmed"
	esummary_options = "&usehistory=y&retmode=xml&version=2.0"
	
	print("Retrieving records for %s PMIDs..." % len(pmids))
	
	try:
		#POST using epost first, with all PMIDs
		idstring = ",".join(pmids)
		queryURL = baseURL + epost
		args = urllib.urlencode({"db":"pubmed","id":idstring})
		response = urllib2.urlopen(queryURL, args)
		
		response_text = (response.read()).splitlines()
		
		webenv_value = (response_text[3].strip())[8:-9]
		webenv = "&WebEnv=" + webenv_value
		querykey_value = (response_text[2].strip())[10:-11]
		querykey = "&query_key=" + querykey_value
		
		#Now we have a search on the history server, so get batches
		i = 0
		
		pmid_count = len(pmids)
		
		pbar = tqdm(total=pmid_count)
		
		with open(filename, "ab") as out_file:
			chunk = 1048576
			batch_size = 1000 #Should be < 100,000
				#see https://www.ncbi.nlm.nih.gov/books/NBK25499/
				#Also if batch is bad (e.g. timeout) 
				#we lose the whole thing. So keep it fairly small.
				#Note that retmax is the total record count we want,
				#not an index value
			last_batch = False
			while True:
				retstart = "&retstart=" + str(i)
				if i + batch_size > pmid_count:
					batch_size = pmid_count - i
					last_batch = True
				retmax = "&retmax=" + str(batch_size)
				queryURL = baseURL + esummary + querykey + webenv \
							+ retstart + retmax + esummary_options
				
				response = urllib2.urlopen(queryURL)
				
				while True:
					data = (response.read(chunk)) #Read one Mb at a time
					out_file.write(data)
					if not data:
						break
								
				i = i + batch_size
				pbar.update(batch_size)
				if last_batch:
					break
					
		pbar.close()
		print("Retreived records and saved to %s." % filename)
			
	except urllib2.HTTPError as e:
		print("Couldn't complete PubMed search: %s" % e)

def find_citation_counts(pmids):
	#Given a list of PMIDs, return counts of PMC citation counts
	#by publication and by ID.
	#Also produces two files:
	#the raw output of the Pubmed search in XML 
	#and a file containing a PMID, its corresponding
	#citation count, and its publication, one per line.
	
	def parse_file_for_cites(pmids, searchfilename, countsfilename):
		#Parse PubMed XML file for citation counts
		#Just return records for searched PMIDs
		#Also write cite counts to file
		
		counts_by_pmid = {} #Tuples of citation counts and pubs,
						#PMIDs are keys
		
		print("Parsing citation counts for %s PMIDs..." % len(pmids))
		
		pmids = set(pmids)
		
		i = 0
		
		pbar = tqdm(total=len(pmids))
		
		with open(searchfilename) as searchfile:
			get_info = False
			for line in searchfile:
				if not line.strip(): #Skip blank lines
					continue
				splitline = line.split("<")
				if len(splitline) < 2: #Skip any other non-field lines
					#This may happen if download(s) fail
					continue
				if splitline[1][0:19] == "DocumentSummary uid":
					this_pmid = (splitline[1].split("\""))[1]
					if this_pmid in pmids:
						get_info = True
				if splitline[1][0:7] == "Source>" and get_info:
					this_pub = (splitline[1].split(">"))[1]
				if splitline[1][0:12] == "PmcRefCount>" and get_info:
					this_count = (splitline[1].split(">"))[1]
					counts_by_pmid[this_pmid] = (this_count, this_pub)
					get_info = False
					i = i +1
					pbar.update(1)
					if i == len(pmids):
						break
		
		pbar.close()
		
		#Convert to just counts	
		for pmid in counts_by_pmid:
			raw_cite_counts[pmid] = counts_by_pmid[pmid][0]
		
		print("Found citation counts for %s records." \
				% len(raw_cite_counts))
		
		#Get counts of counts
		#Discretize to make counts more informative
		#Write the counts to file, too
		#Append counts as we may be searching more than one
		#set of IDs
		with open(countsfilename, 'ab') as countsfile:
			for pmid in counts_by_pmid:
				count_num = int(counts_by_pmid[pmid][0])
	
				if count_num == 0:
					count_num_str = str(count_num)
				elif count_num in [1,2,3,4,5]:
					count_num_str = "1-5"
				elif count_num in [6,7,8,9,10]:
					count_num_str = "6-10"
				elif count_num > 10:
					count_num_str = ">10"
				
				pub_name = counts_by_pmid[pmid][1]
				if pub_name not in counts:
					counts[pub_name] = {}
				
				if count_num_str not in counts[pub_name]:
					counts[pub_name][count_num_str] = 1
				else:
					counts[pub_name][count_num_str] = counts[pub_name][count_num_str] +1
					
				outstring = "%s\t%s\t%s\n" % (pmid, count_num, pub_name)
				countsfile.write(outstring)
		
				
		return counts, raw_cite_counts
	
	counts = {} #Counts of citation counts by publication
				#A dict of dicts, with pub name as first key,
				#group as second key, and group count as value
				#e.g. {"Med Journal" :{"6-10": 5}}
	raw_cite_counts = {} #PMIDs are keys, cite counts are values

	outfiledir = "output"
	searchfilename = "searched_documents.xml"
	countsfilename = "citation_counts.tsv"
	
	if len(pmids) > 0:
		print("Searching for citation counts for %s records." %
				len(pmids))
	else:
		print("No IDs provided to find citation counts for.")
		return counts, raw_cite_counts, countsfilename
		
	if not os.path.isdir(outfiledir):
		os.mkdir(outfiledir)
	os.chdir(outfiledir)
	
	#We may already have a file of searched records.
	#Parse it first, then retrieve only records not present in the file
	#already.
	if os.path.isfile(searchfilename):
		print("Searching local file first...")
		counts, raw_cite_counts = \
			parse_file_for_cites(pmids, searchfilename, countsfilename)
		
		print("Checking to see if all PMIDs have values...")
		orig_pmids = set(pmids)
		found_pmids = set(raw_cite_counts.keys())
		remaining_pmids = orig_pmids.difference(found_pmids)
		pmids = list(remaining_pmids)
	else:
		print("No local file found.")
		
	if len(pmids) > 0: #Go get records from PubMed if we still need any
		print("Retrieving records for PMIDs from PubMed...")
		get_file_and_append(pmids, searchfilename)
		
		more_counts, more_raw_counts = \
			parse_file_for_cites(pmids, searchfilename, countsfilename)
				
		#Now update the lists with any new results
		counts.update(more_counts)
		raw_cite_counts.update(more_raw_counts)

	os.chdir("..")
	
	return counts, raw_cite_counts, countsfilename
