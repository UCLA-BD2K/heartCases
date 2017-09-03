# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:30:07 2017

@author: Jack
"""
PMID_regex = r'PMID- (\d+)'

import re, os, sys
import argparse
import pickle
import pip #Just using it to check on installed packages
installed = pip.get_installed_distributions()
install_list = [item.project_name for item in installed]
need_install = []
for name in ['nltk', 'sklearn', 'tqdm', 'beautifulsoup4']:
	if name not in install_list:
		need_install.append(name)
if len(need_install) > 0:
	sys.exit('Please install these Python packages first:\n%s' 
				% '\n'.join(need_install))
import nltk
import urllib
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
from nltk import sent_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression

def get_features_from_medline(medline):
    """
    This functions gives the features to classify whether or not a medline file 
    describes a single or multiple patients. 
    """
    # Regular Expressions for extracting abstract and MeSH terms from a MEDLINE file in a string format
    abstract_regex = r'AB[\s-]+([\s\S]*?)(CI|FAU|AD)'
    mesh_terms_regex = r'MH[\s-]+([\w\-\*\,\/\ ]+)\n'
    
    abstract = re.search(abstract_regex, medline)
    abstract_text = ''
    if abstract:    # If abstract found
        abstract_text = abstract.group(1)
        abstract_text = re.sub(r'\n', ' ', abstract_text)
        abstract_text = re.sub(r'\s{2,}', ' ', abstract_text)
    # List of MeSH terms
    mesh_terms = re.findall(mesh_terms_regex, medline)
    
    # Regular expression for finding phrases that are indicative of a multi-patient case report
    multi_case_regex = r'((T|t)wo|(T|t)hree|(F|f)our|(F|f)ive|(S|s)ix|(S|s)even|(E|e)ight|(N|n)ine|(T|t)en|[\d]+) ((C|c)ase|(P|p)atient|(P|p)ersons|(P|p)eople|(M|m)en|(W|w)omen)'
    features = {}
    if abstract_text:
        abstract_sentences = sent_tokenize(abstract_text)
        first_sentence_indicator = len(re.findall(multi_case_regex, abstract_sentences[0]))
        # Remove the first sentence because first sentence is a general introduction that may include words that indicate multiple patients
        del abstract_sentences[0]
        abstract = ' '.join(abstract_sentences)
        # The following strings indicate multiple patients in a case report
        multi_patient_words = ['patients', 'cases', 'Case 1', 'case 1', 
                               'Case 2', 'case 2', 'Patient 1', 'patient 1', 
                               'Patient 2', 'patient 2', 'Patient A', 'patient A', 
                               'Patient B', 'patient B', 'Women', 'Men']
        # Count up all the times these multi patient indicator words appear in the abstract
        multi_indicator = [abstract.count(each) for each in multi_patient_words]
        multi_indicator2 = len(re.findall(multi_case_regex, abstract))
        # Find the total count of all multi-patient indicator words
        features['multi_indicator'] = sum(multi_indicator) + first_sentence_indicator + multi_indicator2
        features['single_indicator'] = 'a case' in abstract

    else:
        features['multi_indicator'] = 0
    sex_terms = 'Male' in mesh_terms
    sex_terms += 'Female' in mesh_terms    
    features['sex_terms'] = sex_terms
    # The following MeSH terms also indicate whether or not there are multiple patients in a case report
    age_mesh_terms = ['Adult', 'Aged', 'Aged, 80 and over', 'Child', 'Infant', 'Infant, Newborn', 'Middle Aged']
    features['multiple_age_groups'] = len([each for each in age_mesh_terms if each in mesh_terms]) > 1
    
    animals = 'Animals' in mesh_terms
    species = 'human'
    if animals:
        species = 'animal'
    
    return (species, features)

def get_features_from_pmid(pmid):
    """
    This function is the same as get_features_from_medline but receives a 
    PubMed ID and scrapes the PubMed website for the abstract and MeSH terms
    """
    link = 'https://www.ncbi.nlm.nih.gov/pubmed/?term=' + str(pmid)
    page_data = urllib.request.urlopen(link)
    soup = bs(page_data, 'html.parser')
    abstract_text = ''
    mesh_terms = []
    try:
        abstract_text = [tag.abstracttext.text for tag in soup.select('p') if tag.abstracttext][0]
        mesh_terms = [term.text for term in soup.find_all('a', attrs={'alsec':'mesh'})]
    except:
        pass
    
    multi_case_regex = r'((T|t)wo|(T|t)hree|(F|f)our|(F|f)ive|(S|s)ix|(S|s)even|(E|e)ight|(N|n)ine|(T|t)en|[\d]+) ((C|c)ase|(P|p)atient|(P|p)ersons|(P|p)eople|(M|m)en|(W|w)omen)'
    features = {}
    
    if abstract_text:
        abstract_sentences = sent_tokenize(abstract_text)
        first_sentence_indicator = len(re.findall(multi_case_regex, abstract_sentences[0]))
        #Remove the first sentence because first sentence is a general introduction that may include words that indicate multiple patients
        del abstract_sentences[0]
        abstract = ' '.join(abstract_sentences)
        multi_patient_words = ['patients', 'cases', 'Case 1', 'case 1', 
                               'Case 2', 'case 2', 'Patient 1', 'patient 1', 
                               'Patient 2', 'patient 2', 'Patient A', 'patient A', 
                               'Patient B', 'patient B', 'Women', 'Men']
        multi_indicator = [abstract.count(each) for each in multi_patient_words]
        multi_indicator2 = len(re.findall(multi_case_regex, abstract))
        features['multi_indicator'] = sum(multi_indicator) + first_sentence_indicator + multi_indicator2
        features['single_indicator'] = 'a case' in abstract
    else:
        features['multi_indicator'] = 0
        features['single_indicator'] = False
    sex_terms = 'Male' in mesh_terms
    sex_terms += 'Female' in mesh_terms
    features['sex_terms'] = sex_terms
    age_mesh_terms = ['Adult', 'Aged', 'Child', 'Infant', 'Middle Aged']
    features['multiple_age_groups'] = len([each for each in age_mesh_terms if each in mesh_terms]) > 1
    
    animals = 'Animals' in mesh_terms
    species = 'human'
    if animals:
        species = 'animal'
    
    return (species, features)

with open('logistic_regression.pickle', 'rb') as lrc:
    lr_classifier = pickle.load(lrc)
lrc.close()

def ClassifyMEDLINE(medline):
    """
    This function classifies whether or not a medline file describes a single 
    patient or multiple patients
    """
    species, features = get_features_from_medline(medline)
    if species == 'animal':
        return 'This case report is about animals'
    return lr_classifier.classify(features)

def ClassifyPMID(pmid):
    """
    This function classifies whether or not a PubMed ID is the ID of a single 
    or multiple patient case report
    """
    species, features = get_features_from_medline(pmid)
    if species == 'animal':
        return 'This case report is about animals'
    return lr_classifier.classify(features)

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help='FOLDER is the name of a folder containing '
                                    'individual .txt files of case reports in MEDLINE '
                                    'format.')
    parser.add_argument('--pmids', help='PMIDS is the name of a .txt file containing '
                                    'the PubMed IDs of case reports to classify.')
    parser.add_argument('--medline', help='MEDLINE is the name of a .txt file '
                                    'containing multiple case reports in MEDLINE format.')
    
    try:
        args = parser.parse_args()
    except:
        sys.exit()
    
    return args

def main():
    args = parse_args()
    files_folder = str(args.folder)
    if args.folder:
        try:
            file_list = os.listdir(files_folder)
        except:
            sys.exit('No such folder found or no files inside folder, exiting.')
        pmid_classification = []
        os.chdir(files_folder)
        for each in tqdm(file_list):
            medline_file = open(each, 'r')
            medline_text = medline_file.read()
            pmid_classification.append(re.search(PMID_regex, medline_text).group(1) + 
                                       '\t' + ClassifyMEDLINE(medline_text) + '\n')
            medline_file.close()
        os.chdir('..')
        with open('Classification_Results.txt', 'w') as cr:
            for result in pmid_classification:
                cr.write(result)
        cr.close()
    
    elif args.pmids:
        try:
            pmid_file = open(str(args.pmids), 'r')
            pubmed_ids = pmid_file.read().split()
            pmid_file.close()
        except:
            sys.exit('No valid file found, exiting.')
        pmid_classification = []
        for each in tqdm(pubmed_ids):
            pmid_classification.append(each + '\t' + ClassifyPMID(each) + '\n')
        with open('Classification_Results.txt', 'w') as cr:
            for result in pmid_classification:
                cr.write(result)
        cr.close()
    
    elif args.medline:
        try:
            medline_file = open(str(args.medline), 'r')
            case_reports = medline_file.read()
            medline_files = case_reports.split('\n\n')
        except:
            sys.exit('No valid file found, exiting.')
        pmid_classification = []
        for each in tqdm(medline_files):
            pmid_classification.append(re.search(PMID_regex, each).group(1) + 
                                       '\t' + ClassifyMEDLINE(each) + '\n')
        with open('Classification_Results.txt', 'w') as cr:
            for result in pmid_classification:
                cr.write(result)
        cr.close()
    else:
        sys.exit('Please enter --folder FOLDERNAME, --pmid PMIDS, or --medline MEDLINES '
                 'after caseReport_classify.py')
    

if __name__ == '__main__':
    sys.exit(main())