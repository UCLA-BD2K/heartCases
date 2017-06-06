#!/usr/bin/python
#heartCases_help.py
'''
heartCases is a medical language processing system for case reports 
involving cardiovascular disease (CVD).

This part of the system includes helper functions.

'''

import os, sys, urllib, urllib2
from tqdm import *

def get_file_and_append(pmids, filename):
	#Given a list of PMIDs, download PubMed records using eutils
	#and history server. Append to the file if it already exists.
	
	baseURL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
	epost = "epost.fcgi"
	esummary = "esummary.fcgi?db=pubmed"
	esummary_options = "&usehistory=y&retmode=xml&version=2.0"
	batch_size = 500
	
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
		
		pbar = tqdm(total=len(pmids))
		
		while i <= len(pmids):
			retstart = "&retstart=" + str(i)
			retmax = "&retmax=" + str(i + batch_size)
			queryURL = baseURL + esummary + querykey + webenv \
						+ retstart + retmax + esummary_options
			
			response = urllib2.urlopen(queryURL)
			
			with open(filename, "a+") as out_file:
				chunk = 1048576
				while 1:
					data = (response.read(chunk)) #Read one Mb at a time
					out_file.write(data)
					if not data:
						break			
				i = i + batch_size
				pbar.update(batch_size)
		
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
			for line in searchfile:
				if not line.strip(): #Skip blank lines
					continue
				splitline = line.split("<")
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
		
		#Get counts of counts
		#Discretize to make counts more informative
		#Write the counts to file, too
		#Append counts as we may be searching more than one
		#set of IDs
		with open(countsfilename, 'a+') as countsfile:
			for pmid in counts_by_pmid:
				count_num = int(counts_by_pmid[pmid][0])
	
				if count_num == 0:
					count_num_str = str(count_num)
				elif count_num in [1,2,3,4,5]:
					count_num_str = "1-5"
				elif count_num in [6,7,8,9,10]:
					count_num_str = "6-10"
				elif count_num > 9:
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
		
		print("\nRetrieved citation counts for %s records." \
				% len(pmids))
				
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
		print("Searching for citation counts for %s records in total." %
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
	
	if len(pmids) > 0: #Go get records from PubMed if we still need any
		print("Retrieving records for PMIDs not in local file "
				"or no local file found...")
		get_file_and_append(pmids, searchfilename)
		
		more_counts, more_raw_counts = \
			parse_file_for_cites(pmids, searchfilename, countsfilename)
				
		#Now update the lists with any new results
		counts.update(more_counts)
		raw_cite_counts.update(more_raw_counts) 

	os.chdir("..")
	
	return counts, raw_cite_counts, countsfilename
