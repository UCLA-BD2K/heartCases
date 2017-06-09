#!/usr/bin/python
#heartCases_learn.py
'''
heartCases is a medical language processing system for case reports 
involving cardiovascular disease (CVD).

This part of the system is intended for learning from the labels and
features identified in a set of case reports processed by
heartCases_read.

Features are primarily sentence-level labels provided by the 
heartCases_read NER and relationship labelling. This module then uses
a convolutional neural network (CNN) for further sentence-level
classification, followed by additional feature learning to identify
document-level features.

The initial sentence classification relies upon word embeddings. These
may be pre-trained but should preferentially be trained on medical
language.

Uses gensim for now to produce word2vec word vectors.

'''
__author__= "Harry Caufield"
__email__ = "j.harry.caufield@gmail.com"

import argparse, sys
from gensim.models.word2vec import *

#Constants and Options

#Functions

def build_word_vectors(inputfilename):
	#Use gensim word2vec implementation to build word vectors
	sentences = LineSentence(inputfilename)
	model = Word2Vec(sentences, size=100, window=5, min_count=5, 
						workers=4, hs=1, negative=0)
	model.save("model")
def parse_args():
	#Parse command line arguments for heartCases_learn
	
	parser = argparse.ArgumentParser()
	parser.add_argument('vtrain', help="the name of a file to use "
						"for building word vectors.")
	parser.add_argument('--verbose', help="if TRUE, provide verbose output")
	
	try:
		args = parser.parse_args()
	except:
		sys.exit()
	
	return args

#Main
def main():
	#Set up parser
	args = parse_args()
	
	vtrain_filename = args.vtrain
	print("Using this file to build word vectors: %s" % vtrain_filename)
	
	print("Processing...")
	#No preprocessing here yet.
	
	print("Building word vectors...")
	build_word_vectors(vtrain_filename)

if __name__ == "__main__":
	sys.exit(main())
