# heartCases
A medical language processing system for case reports involving cardiovascular disease.

(Work in progress.)

This system includes several different modules.

## heartCases_read.py

This part of the system is intended for parsing MEDLINE format files 
and specifically isolating those relevant to cardiovascular disease (CVD).

This script attempts to expand on existing MeSH annotations by performing
tag classification with MeSH terms and adding terms to records where
appropriate. These terms can optionally include just those used to search
records (e.g., if only terms related to heart disease are provided,
a classifier will be trained only to add those terms when missing.)
Similar approaches have been employed by [Huang et al. (2011) JAMIA.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3168302/)
(Huang et al. used a k-nearest neighbors approach to get related articles
and their highest-ranking MeSH terms. They achieved ~71% recall on 
average.)

Both previously present and newly added annotations are used to further annotate records with relevant ICD-10 disease codes.

A separate classifier is then used to determine medically-relevant content within abstracts, including demographic details, symptoms, and lab values, among other features.

### Requirements 
Requires [numpy](http://www.numpy.org/), [nltk](http://www.nltk.org/), and [scikit-learn](http://scikit-learn.org/stable/).

Uses the Disease Ontology project database; see
http://www.disease-ontology.org/ or [Kibbe et al. (2015) NAR.](https://www.ncbi.nlm.nih.gov/pubmed/25348409)

Uses the [2017 MeSH Data Files provided by the NIH NLM](https://www.nlm.nih.gov/mesh/filelist.html).
These files are used without modification.

Uses the [2017 SPECIALIST Lexicon](https://lexsrv3.nlm.nih.gov/Specialist/Summary/lexicon.html).
See SPECIALIST.txt for terms and conditions regarding the use of the SPECIALIST NLP tools (don't worry, they're short).

The three data sets listed above are downloaded if not present locally. All together, these sets require ~1.8 Gb of disk space, most of which is for SPECIALIST. 

### Usage
Run as:
`python heartCases_read.py`

#### Input
Text files containing literature references in MEDLINE format.
Files to process should be placed in the "input" folder.

Alternatively, provide the input file as an argument:
  `--inputfile INPUT_FILE_NAME`

where INPUT_FILE_NAME is a text file containing one or more documents in MEDLINE format.

Input may also be provided as a text file of PubMed IDs, with one ID per line, using the argument:
  `--pmids INPUT_FILE_OF_PMIDS`


#### Output
Outputs are saved to the "output" folder at the end of each run.
These include three files containing the name of the input
(or "medline_entries_" if more than one input file is provided)
appended with the number of documents contained in the output
and one of the following:
* _relabeled.txt - all documents provided in the input file(s) which
	match the given seach space (in this case, cardiovascular disease)
	with additional MeSH terms and ICD-10 annotations, where possible.
* _sentences.txt - all document titles, PMIDs, and all abstract text
	labeled by sentence with topics each may contain.
* _topics.txt - all document titles, PMIDs, and their corresponding
	contents, listed by topic heading.
	
Counts of matched MeSH terms, journals, publication years,
and other details are also provided.
