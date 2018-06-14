import os
import re
import pickle
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from gensim.models.word2vec import Word2Vec

model = Word2Vec.load('full_model')

with open('features.pickle', 'rb') as f:
    features = pickle.load(f)
f.close()

with open('labels.pickle', 'rb') as g:
    labels = pickle.load(g)
g.close()


def keywords(sentence):
    keyterms = {'died': 0,
             'expired': 0,
             'dead': 0,
             'death': 0,
             'asymptomatic': 0,
             'disease-free': 0,
             'symptom-free': 0,
             'no-recurrence': 0,
             'uneventful': 0,
             'recovered': 0,
             'stable': 0,
             'survived': 0,
             'discharged': 0,
             'treated': 0,
             'remained': 0,
             'complete': 0,
             'completely': 0,
             'resolved': 0,
             'improvement': 0,
             'improved': 0,
             'cured': 0,
             'resolved': 0,
             'free': 0,
             'alive': 0,
             'restored': 0,
             'progressed': 0,
             'successfully': 0,
             'unremarkable': 0}
    words = word_tokenize(sentence)
    for i, each in enumerate(words):
        if each.lower().strip() in keyterms:
            keyterms[each.lower().strip()] += 1
        elif each.lower().strip() in ['recurrent', 'recurrence']:
            if i-3 >= 0:
                if words[i-3] in ['no', 'without', 'devoid', 'absent']:
                    keyterms['no-recurrence'] += 1
            if i-2 >= 0:
                if words[i-2] in ['no', 'without', 'devoid', 'absent']:
                    keyterms['no-recurrence'] += 1
            if i-1 >= 0:
                if words[i-1] in ['no', 'without', 'devoid', 'absent']:
                    keyterms['no-recurrence'] += 1
    return keyterms

def getKeySentences(report):
    sentences = sent_tokenize(report.lower())
    patient = []
    heshe = []
    hisher = []
    numbers = []
    case = []
    for sentence in sentences:
        if 'patient ' in sentence or 'patient\'' in sentence:
            patient.append(sentence)
        elif ' he ' in sentence or ',he ' in sentence or 'she ' in sentence:
            heshe.append(sentence)
        elif ' his ' in sentence or ',his ' in sentence or 'her ' in sentence:
            hisher.append(sentence)
        elif re.search('one |two |three |four |five |six |seven |eight |nine |ten ', sentence):
            numbers.append(sentence)
        elif 'case ' in sentence:
            case.append(sentence)
    return [patient[::-1], heshe[::-1], hisher[::-1], numbers[::-1], case[::-1]]

def get_features(report):
    keep_updating = True
    key_sentences = getKeySentences(report)
    final_features = keywords('')
    for each in key_sentences:
        for sentence in each:
            key_words = keywords(sentence)
            if key_words['died'] > 0 or key_words['expired'] > 0:
                return list(key_words.values())
            if sum(key_words.values()) > 0 and keep_updating:
                final_features = key_words
                keep_updating = False
    return list(final_features.values())

dtc = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, 
                       min_samples_split=2, min_samples_leaf=1, 
                       min_weight_fraction_leaf=0.0, max_features=None, 
                       random_state=None, max_leaf_nodes=None, 
                       min_impurity_decrease=0.0, min_impurity_split=None, 
                       class_weight=None, presort=False)
dtc.fit(features, labels)
accuracy_score([each.index(1) for each in labels], [each.argmax() for each in dtc.predict(features)])

test_files = os.listdir('full_texts/test_set2')
os.chdir('full_texts/test_set2')
test_set = []
y = []
for each in test_files:
    if 'D' in each:
        y.append(2)
    elif 'S' in each:
        y.append(1)
    else:
        y.append(3)
    with open(each, 'r', encoding='utf-8') as f:
        test = f.read()
        test_set.append(test)
    f.close()

test_features = [get_features(each) for each in test_set]
y_hat = dtc.predict(test_features)
y_hat = [int(prediction.argmax()+1) for prediction in y_hat]
accuracy_score(y_hat, y)
