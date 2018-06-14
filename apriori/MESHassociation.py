
PMID_regex = r'PMID[- ]*?(\d+)'
abstract_regex = r'AB  - ([\s\S]*?)FAU'
mesh_terms_regex = r'\nMH[\s-]+([\w\-\*\,\/\ ]+)'

import re
import statistics
from apyori import apriori

def getMESHterms(medline, ignoreTerms = {}):
    # List of MeSH terms
    mesh_terms = re.findall(mesh_terms_regex, medline)
    toRemove = []
    for term in mesh_terms:
        if term in ignoreTerms:
            toRemove.append(term)
    for each in toRemove:
        mesh_terms.remove(each)
    for i in range(0, len(mesh_terms)):
        if re.search('(.+?)[\/,]', mesh_terms[i]):
            mesh_terms[i] = re.findall('(.+?)[\/,]', mesh_terms[i])[0]
    return mesh_terms

class PubMedCaseReport:
    def __init__(self, casereport, mesh=False, ab=False):
        self.pmid = int(re.search(PMID_regex, casereport)[1])
        self.MESH_terms = []
        if mesh:
            self.MESH_terms = getMESHterms(casereport)
        self.abstract = None
        if ab and re.search(abstract_regex, casereport):
            abstract = re.search(abstract_regex, casereport)[1]
            abstract = re.sub(r'\n', ' ', abstract)
            self.abstract = re.sub(r'\s{2,}', ' ', abstract)

with open('ACCR.txt', 'r', encoding="utf8") as f:
    data = f.read()
f.close()

caseReports = data.split('\n\n')

mesh_terms = [getMESHterms(casereport) for casereport in caseReports]
pubmed_classes = [PubMedCaseReport(casereport, True) for casereport in caseReports]

term_counts = {}
for terms in mesh_terms:
    for term in terms:
        if term in term_counts:
            term_counts[term] += 1
        else:
            term_counts[term] = 1

count_distribution = []
for each in term_counts:
    count_distribution.append(term_counts[each])

avg_appearance = statistics.mean(count_distribution)
std_dev = statistics.stdev(count_distribution)
extreme_occurrance_terms = {}
for term in term_counts:
    if term_counts[term] > avg_appearance + 5*std_dev or term_counts[term] < avg_appearance:
        extreme_occurrance_terms[term] = term_counts[term]

filtered_mesh_terms = []
for report in caseReports:
    if getMESHterms(report, extreme_occurrance_terms):
        filtered_mesh_terms.append(getMESHterms(report, extreme_occurrance_terms))

results = list(apriori(filtered_mesh_terms, min_support = .005, min_lift = 1.5))

for result in results:
    print(result.items)

import numpy as np
import matplotlib.pyplot as plt

n, bins, patches = plt.hist(np.log(count_distribution), 100, facecolor='blue', alpha=0.5, range=(0, 15))
plt.show()

groups_PMIDs = []
for each in results:
    associated_pmids = []
    for pubmed in pubmed_classes:
        flag = True
        for term in each.items:
            if term not in pubmed.MESH_terms:
                flag = False
        if flag:
            associated_pmids.append(pubmed.pmid)
    groups_PMIDs.append((each.items, associated_pmids))

frequency_counts = {}
for each in results:
    for term in each.items:
        if term in frequency_counts:
            frequency_counts[term] += 1
        else:
            frequency_counts[term] = 1


#with open('ACCR_MESH_groups.txt', 'w') as w:
#    for result in results:
#        w.write(','.join(result.items))
#        w.write('\n')
#w.close()