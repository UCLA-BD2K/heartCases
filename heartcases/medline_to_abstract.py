#!/usr/bin/python
#medline_to_abstract.py
'''
Given a text file containing literature records in MEDLINE format,
 provides a file PMIDs, PMC IDs, and abstract text only.
'''
__author__= "Harry Caufield"
__email__ = "j.harry.caufield@gmail.com"

import argparse, sys

#Classes
class Record(dict):
	'''
	Just establishes that the record is a dict
	'''
	
#Functions

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
		
def parse_args():
	'''Parse command line arguments'''
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputfile', help="name of a text file containing "
						"MEDLINE records")
	
	try:
		args = parser.parse_args()
	except:
		sys.exit()
	
	return args

##Main
def main():
	
	#Set up parser
	args = parse_args()
	
	if args.inputfile:
		medline_file = args.inputfile
	else:
		sys.exit("No input file provided.")
		
	print("\nLoading %s..." % medline_file)
	outfilename = (medline_file)[:-4] + "_abstracts.txt"
	
	with open(medline_file) as this_medline_file:
		print("Reading entries...")
		records = medline_parse(this_medline_file)
		outfile = open(outfilename, 'w')
		for record in records:
			pmid = record['PMID']
			try:
				pmc_id = record['PMC']
			except KeyError:
				pmc_id = "NA"
			try:
				abst = record['AB']
				have_abst = True
			except KeyError:
				have_abst = False
			if have_abst:
				outstring = "%s\t%s\t%s\n" % (pmid, pmc_id, abst)
				outfile.write(outstring)
				
	print("Wrote output to %s." % outfilename)
			
if __name__ == "__main__":
	sys.exit(main())
