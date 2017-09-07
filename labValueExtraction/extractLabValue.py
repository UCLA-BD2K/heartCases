# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:25:06 2017

@author: Jack
"""


value_regex = r'([\d.,]*[\/]?[\d.,]+)'  #regular expression to extract the numeric value of a lab value
lab_value_regex = '[^A-Za-z\d]([\d]+[\d.,\/]* ?[α-ωΑ-Ω\w\/]+( ?[A-Zα-ωΑ-Ω][a-zα-ωΑ-Ω]+)?( ?per \w+)?)'  #regular expression to find potential lab values

import os, re, sys
import logging
import pickle
import argparse
import pip #Just using it to check on installed packages
installed = pip.get_installed_distributions()
install_list = [item.project_name for item in installed]
need_install = []
for name in ["numpy", "nltk", "scipy", "gensim", "tqdm", "beautifulsoup4"]:
	if name not in install_list:
		need_install.append(name)
if len(need_install) > 0:
	sys.exit("Please install these Python packages first:\n%s" 
				% "\n".join(need_install))

import nltk
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup as bs

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load in pre-configured files
os.chdir('files_to_be_loaded')
model = Word2Vec.load('word_embedding')     #Load in the pre-trained word2vec model
center = np.load('center_of_drugs.npy')    #Load in the vector most similar to the highest number of lab value substances

#Load in the classifier to determine whether or not something is a lab value.
with open("Lab_Value_Classifier.pickle", "rb") as lv_c:
    lab_value_classifier = pickle.load(lv_c)
lv_c.close()

#Load in the classifier to determine which word/s are being measured by the lab value found.
with open("Entity_Classifier.pickle", "rb") as ec:
    entity_classifier = pickle.load(ec)
ec.close()

#A dictionary whose values are used for features in classification
with open('sample_substance_counts.pickle', 'rb') as g:
    sample_substance_counts = pickle.load(g)
g.close()

os.chdir('..')


def xml_to_text(xmlfile):
    """
    This function takes a case report in XML format and extracts the PubMed ID
    and the full text of the case report.
    """
    text = open(xmlfile).read()
    soup = bs(text, 'xml')
    try:
        fulltext = soup.select('p')
        sentences = [x.text.strip() for x in fulltext]
        case_report_text = ' '.join(sentences)
        case_report_text = re.sub('\[[\S]+\]', '', case_report_text)
        case_report_text = re.sub('[\s]{2,}', ' ', case_report_text)
        case_report_text = re.sub('This is an Open Access article distributed '
                                  'under the terms of the Creative Commons Attribution '
                                  'License \(\), which permits unrestricted use, '
                                  'distribution, and reproduction in any medium, '
                                  'provided the original work is properly cited\. ', 
                                  '', case_report_text)
        if case_report_text:
            pmid = soup.select('article-id')[0].text
            return {'PMID': pmid, 'text': case_report_text}
    except IndexError:
        pass

def classification_features(word):
    """
    Features used for classifying lab values.
    """
    features = {}
    value = re.findall(value_regex, word)[0]
    unit = word.replace(value, '')
    different_words = unit.replace('/', ' ').split()
    features['starts_with_num'] = RepresentsNum(word[0])
    features['has_divison'] = '/' in unit or 'per' in unit
    features['ends_with_capital'] = word[-1] == word[-1].upper()
    features['keyword'] = 'none'
    if different_words:
        features['keyword'] = different_words[0]
    return features

#Stop words (the, and, to, of, etc.) to remove from being considered as substances
stop_words = set(stopwords.words('english'))

def RepresentsNum(s):
    """
    Determines whether or not a string represents a pure numeric value.
    
    """
    try:
        float(s.replace('/', '').replace(',', '')) #remove any slashes and commas
        return True
    except ValueError:
        return False

def get_phrases(sentence):
    """
    Separates a sentence into phrases using separators such as semicolons
    (if semicolons are found), commas (if no semicolons are found), and 'and'.
    """
    if ';' in sentence:
        return sentence.replace(' and ', ';').split(';')
    else:
        return sentence.replace(' and ', ', ').split(', ')

def get_word_vector(word):
    try:
        return model[word]
    except KeyError:
        return None

def features_for_training(value, unit):
    """
    Features used for classifying whether or not a word is the entity being 
    measured by a specific lab value.
    """
    features = {}
    if value.lower() in sample_substance_counts:
        features['count'] = sample_substance_counts[value.lower()]
    else:
        features['count'] = 0
    features['similarity_to_substance_center'] = 1 - cosine(model[value], center)
    features['similarity_to_unit'] = model.wv.similarity(value, unit)
    return features

def most_similar_to_substance_center(word_list):
    """
    Finds the most similar word to the substance_center vector from a list of words.
    
    "word_list" is a list of words that contains the most similar word to substance_center.
    
    """
    similarities = np.array([])
    for each in word_list:
        try:
            similarities = np.append(similarities, 1-cosine(center, model[each]))
        except KeyError:
            similarities = np.append(similarities, np.array([0]))
    return word_list[similarities.argmax()]

def get_lab_values(case_report):
    """
    Returns a list of lab values, their units, and the substances they describe.
    Prints "No lab values found in report" if no lab values are found.
    
    Input "case_report" is an article in a string format or a dictionary with 
    key 'PMID' containing the PubMed ID and key 'text' containing the full_text 
    in string format.
    
    """
    is_dict = isinstance(case_report, dict)     # Check the format of the input
    if is_dict:
        pmid = case_report['PMID']
        report = case_report['text']
    else:
        report = case_report
    
    lab_values = []     # Initialize the object to return
    sentences = [sentence for sentence in sent_tokenize(report) if len(sentence) > 5]   # Split up article into sentences that contain at least 6 characters.
    for sentence in sentences:  # Split case report into sentences, split sentences into phrases, and search for lab values in each phrase
        phrases = get_phrases(sentence)
        for phrase in phrases:
            lv = [value[0] for value in re.findall(lab_value_regex, phrase) if 
                  lab_value_classifier.classify(classification_features(value[0])) == 'lab value']
            
            if lv:  # Do the following if a lab value is found withing a phrase.
                lab_value = lv[0]
                # Narrow down the possible words for what the lab value is measuring
                phrase_words = [pos[0] for pos in nltk.pos_tag(word_tokenize(phrase)) if 
                                (pos[1] == 'NN' or pos[1] == 'NNS' or pos[1] == 'NNP' or 
                                 pos[1] == 'JJ' or pos[1] == 'RBR') and 
                                 len(pos[0]) > 1 and 
                                 pos[0] not in lab_value]
                # Narrow down further into words that have an embedding
                phrase_words = [word for word in phrase_words if 
                                get_word_vector(word) is not None]
                substance = ''
                try:
                    unit = re.sub(value_regex, '', lab_value)   # Find the unit the lab value is measured in
                    check_embedding_exists = model[unit.strip()]    # Check if the unit has an embedding
                    words_with_features = [(word, features_for_training(word, unit.strip())) for word in phrase_words]
                    words_with_labels = [(word, entity_classifier.classify(feature)) for (word, feature) in 
                                         words_with_features]
                    # End up with the words that are classified as entities being measured by the specific lab value
                    possible_substances = [word[0] for word in words_with_labels if 
                                           word[1] == 'positive']
                    
                    if not possible_substances:     # If no words have over 50% probability of being the measured entity, find the word with the highest probability
                        words_with_probabilities = [(word, entity_classifier.prob_classify(feature).prob('positive')) for 
                                                    (word, feature) in words_with_features]
                        probabilities = [word[1] for word in words_with_probabilities]
                        if probabilities:
                            substance = phrase_words[np.argmax(probabilities)]
                    else:
                        substance = '-'.join(possible_substances)   # Join all possible entities with '-'
                except:     # If there is no word embedding for the lab value unit, then find the word in the phrase most similar to 'center1'
                    if phrase_words:
                        substance = most_similar_to_substance_center(phrase_words)
                    else:
                        substance = ''
                labValue = {'lab value':lab_value, 'measured':substance}
                lab_values.append(labValue)
    if is_dict:
        return (pmid, lab_values)
    else:
        return lab_values

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', help='EXTRACT is the name of a folder containing '
                        'XML files of case reports to extract lab values from.')
    
    try:
        args = parser.parse_args()
    except:
        sys.exit()
    
    return args

#Main
def main():
    args = parse_args()
    if args.extract:
        folder = args.extract
    else:
        sys.exit('No folder entered, please provide a folder containing XML files '
                 'in the form of --extract FOLDER_NAME after extract_lab_values.py')
    
    try:
        file_list = os.listdir(folder)
        check_if_folder_contains_file = file_list[0]
    except:
        sys.exit('No such folder found or no files inside folder, exiting.')
    
    os.chdir(folder)
    xml_files = []
    print('Reading XML files')
    for each in tqdm(file_list):
        xml_files.append(xml_to_text(each))
    xml_files = [each for each in xml_files if each]
    os.chdir('..')
    
    print('Preparing new files for training')
    case_report_sentences = [sent_tokenize(each['text']) for each in tqdm(xml_files)]
    sentences = [sentence for each in case_report_sentences for sentence in each]
    sentence_words = [word_tokenize(sentence) for sentence in tqdm(sentences)]
    model.build_vocab(sentence_words, update=True)
    
    all_lab_values = []
    print('extracting lab values\n')
    for each in tqdm(xml_files):
        all_lab_values.append(get_lab_values(each))
    
    print('writing lab values to file\n')
    with open('Lab Values.txt', 'w', encoding='utf-8') as f:
        for each in all_lab_values:
            f.write('PMID: ' + each[0] + '\n')
            for value in each[1]:
                for key, value in value.items():
                    f.write('%s:\t%s\n' % (key, value))
            f.write('\n\n')
    f.close()

if __name__ == '__main__':
    sys.exit(main())