#!/usr/bin/python
#get_mesh_term.py
'''
Given a text file containing a list of MeSH unique IDs, provides
the corresponding MeSH terms.
Requires the MeSH ontology file AND the MeSH Supplementary Records file,
both in ASCII format (.bin) to be present in the same directory.

If the use_supplementary_terms option is True,
IDs for supplementary terms will be converted to the terms for their
corresponding headings, e.g. C537043 corresponds to the Supplementary
Concept of "Albinism ocular late onset sensorineural deafness" but maps
to the headings "Hearing Loss, Sensorineural" and "Albinism, Ocular".
'''
__author__= "Harry Caufield"
__email__ = "j.harry.caufield@gmail.com"

import re, sys

#Constants and Options
use_supplementary_terms = False

input_filename = "all_mesh_ids_in_DO.txt"
mo_filename = "d2017.bin"
mosr_filename = "c2017.bin"

#Functions
def load_input_list(input_filename):
	'''Gets list of unique MeSH IDs from file.'''
	
	mesh_id_list = []
	
	with open(input_filename) as input_file:
		for line in input_file:
			mesh_id_list.append(line.rstrip())
			
	return mesh_id_list
	
def build_mesh_dict():
	'''Builds dictionary of unique MeSH IDs as keys and their terms
	as values. There may be multiple terms. 
	Also includes IDs for supplementary records.'''
	
	mo_ids = {}	#MeSH IDs (UI in term ontology file) are keys,  
				#terms are values
	
	with open(mo_filename) as mo_file:
		
		for line in mo_file:	#UI is always listed after MH
			if line[0:3] == "MH ":
				#The main MeSH term (heading)
				term = ((line.split("="))[1].strip()).lower()
			elif line[0:3] == "UI ": #This indicates the end of an entry
				clean_id = ((line.split("="))[1].strip())
				if not clean_id in mo_ids:
					mo_ids[clean_id] = [term]
				else:
					mo_ids[clean_id].append(term)
	
	if use_supplementary_terms:
		with open(mosr_filename) as mo_file:
			
			terms = []
			
			for line in mo_file:	#UI is always listed after MH
				if line[0:3] == "HM ":
					#The corresponding header term(s) in the main ontology
					this_term = ((line.split("="))[1].strip()).lower()
					this_term = re.sub('[*]', '', this_term)
					terms.append(this_term)
				elif line[0:3] == "UI ": #This indicates the end of an entry
					clean_id = ((line.split("="))[1].strip())
					if not clean_id in mo_ids:
						mo_ids[clean_id] = terms
					else:
						for term in terms:
							mo_ids[clean_id].append(term)
					terms = []
					
	return mo_ids
	
#Main
def main():
	
	found_count = 0
	not_found_count = 0
	
	mesh_dict = build_mesh_dict()
	ids = load_input_list(input_filename)
	for mesh_id in ids:
		try:
			for term in mesh_dict[mesh_id]:
				print(term)
			found_count = found_count +1
		except KeyError:
			not_found_count = not_found_count +1
	
	print("FOUND: %s ids." % found_count)
	print("NOT FOUND: %s ids." % not_found_count)
	
if __name__ == "__main__":
	sys.exit(main())
