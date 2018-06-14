
import os, re, sys
import textwrap
import nltk
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

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
            return case_report_text
    except IndexError:
        pass


files_ = os.listdir('casereports_xml')
os.chdir('casereports_xml')
full_texts = []
for each in tqdm(files_):
    full_texts.append(xml_to_text(each))
os.chdir('..')

nested_sentences = [sent_tokenize(text) for text in full_texts if text]
sentences = [sentence for text in nested_sentences for sentence in text]
sentence_words = [word_tokenize(sentence) for sentence in sentences]

model = Word2Vec(sentence_words, min_count=5)

os.chdir('full_texts')
labeled_files = os.listdir('labeled')
survived = [each for each in labeled_files if re.search('S', each)]
died = [each for each in labeled_files if re.search('D', each)]
for each in survived:
    labeled_files.remove(each)
for each in died:
    labeled_files.remove(each)
unspecified = labeled_files
os.chdir('labeled')
survived_cases = []
died_cases = []
for each in survived:
    with open(each, 'r', encoding='utf-8') as f:
        report = f.read()
    f.close()
    report = re.sub('\n', ' ', report)
    survived_cases.append(report)

for each in died:
    with open(each, 'r', encoding='utf-8') as f:
        report = f.read()
    f.close()
    report = re.sub('\n', ' ', report)
    died_cases.append(report)

for i, each in enumerate(survived_cases):
    report_sentences = sent_tokenize(each)
    print(survived[i])
    for sentence in report_sentences:
        if 'patient' in sentence:
            print(sentence)
    print('\n')


def main():
    files_ = os.listdir('casereports_xml')
    os.chdir('casereports_xml')
    full_texts = []
    for each in tqdm(files_):
        full_texts.append(xml_to_text(each))
    os.chdir('..')
    os.makedirs('full_texts')
    os.chdir('full_texts')
    for each in tqdm(full_texts):
        if each and 'PMID' in each and 'text' in each:
            with open(each['PMID']+'.txt', 'w', encoding='utf-8') as f:
                lines = textwrap.wrap(each['text'], 140, break_long_words=False)
                for line in lines:
                    f.write(line + '\n')
            f.close()



"""
if __name__ == '__main__':
    sys.exit(main())
"""