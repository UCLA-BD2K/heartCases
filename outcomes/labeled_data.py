import os
import re
import pickle
import pandas as pd
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize

data = pd.read_csv('AllCaseReports.csv', sep='\t')
outcomes = data[['PMID', 'Patient Outcome Assesment']]

true_outcomes = outcomes[outcomes.iloc[:,1].notnull()]
no_outcomes = outcomes[outcomes.iloc[:,1].isnull()]

files = os.listdir('full_texts')
files.remove('labeled')
files = [int(re.sub('[^\d]+', '', each)) for each in files]
labeled_files = [int(re.sub('[^\d]+', '', each)) for each in os.listdir('full_texts/labeled')]

labeled_true_outcomes = [each for each in list(true_outcomes.iloc[:,0])]
labeled_no_outcomes = [each for each in list(no_outcomes.iloc[:,0])]

specified_in_set = []
for each in labeled_true_outcomes:
    if each in labeled_files or each in files:
        specified_in_set.append(str(each)+'.txt')
unspecified_in_set = []
for each in labeled_no_outcomes:
    if each in labeled_files or each in files:
        unspecified_in_set.append(str(each)+'.txt')

labeled_in_folder = pd.DataFrame()
for each in specified_in_set:
    labeled_in_folder = pd.concat([labeled_in_folder, pd.DataFrame(outcomes.loc[outcomes.iloc[:,0] == int(re.sub('[^\d]+','',each)),:])])

unlabeled_in_folder = pd.DataFrame()
for each in unspecified_in_set:
    unlabeled_in_folder = pd.concat([unlabeled_in_folder, pd.DataFrame(outcomes.loc[outcomes.iloc[:,0] == int(re.sub('[^\d]+','',each)),:])])

model = Word2Vec.load('full_model')

full_texts = []
for filename in specified_in_set:
    try:
        with open('full_texts/'+filename, 'r', encoding='utf-8') as f:
            text = f.read()
            text = text.replace('\n', ' ')
            full_texts.append(text)
        f.close()
    except:
        try:
            with open('full_texts/labeled/'+filename, 'r', encoding='utf-8') as f:
                text = f.read()
                text = text.replace('\n', ' ')
                full_texts.append(text)
            f.close()
        except:
            pass

with open('important_sentences.pickle', 'rb') as f:
    important_sentences = pickle.load(f)
f.close()

for i, report in enumerate(full_texts):
    print('REPORT', i)
    if i >= 131:
        sentences = sent_tokenize(report)
        for sentence in sentences:
            print(sentence)
            k = input()
            if k == ' ':
                important_sentences.append(sentence)
            if k == 'break':
                break
        if k == 'break':
            break

with open('important_sentences.pickle', 'wb') as f:
    pickle.dump(important_sentences, f)
f.close()