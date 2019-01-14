#!/usr/bin/env python2
# -*- coding: UTF-8 -*-


# This only works in Python 2 because the farsiNorm script is written in Python 2
# This script takes a text file as input and performs the following 
# preprocessing steps for Farsi:
# 1. get rid of punctuation
# 2. use farsiNorm.py (gets rid of diacritics, ZWNJ, Arabic numerals and others)
# 3. normalize Arabic characters so that initial, middle, and end versions
# of the same letter also appear the same to dumb computers

import sys, re, argparse, urllib, unicodedata
from collections import OrderedDict
from string import punctuation

# Download farsiNorm.py:
urllib.urlretrieve('https://raw.githubusercontent.com/wfeely/farsiNLPTools/master/farsiNorm.py', 'farsiNorm.py')
from farsiNorm import norm_farsi

def farsi_preprocess(file):
	text = []
	nopunc = []
	normed = []
	out = []

	for line in file:
		line = line
		text.append(line.strip())
	#get rid of punctuation
	puncmarks = ['ARABIC COMMA', 'ARABIC QUESTION MARK', 'ARABIC SEMICOLON',\
	 'LEFT-POINTING DOUBLE ANGLE QUOTATION MARK', 'RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK']
	# leave square brackets in because of spoken noise tokens in transcripts
	punc = re.sub('\[', '', punctuation)
	punc = re.sub('\]', '', punc)
	for p in puncmarks:
	 	punc = punc + unicodedata.lookup(p)
	for sentence in text:
		s = ''.join(c for c in sentence.decode('utf-8') if c not in punc)
		nopunc.append(s.encode('utf-8'))
	for sentence in nopunc:
		# normalization with farsiNorm.py
		normed.append(norm_farsi(sentence))
	for sentence in normed:
		# get rid of ZWNJ because farsiNorm.py isn~t perfect in that regard
		sentence = re.sub(u'\u200c'.encode('utf-8'), '', sentence)
		# substitute Arabic letters with Farsi equivalents
		# (the transcription data is somewhat 'contaminated' with these two)
		sentence = re.sub('ك', 'ک', sentence)
		sentence = re.sub('ة', 'ه', sentence)
		# normalizes initial, mid, and end characters into
		# one universally recognised character
		new = unicodedata.normalize('NFKD', sentence.decode('utf-8'))
		out.append(new.encode('utf-8'))

	for i in out:
		print i
	return 0

def main(args):
	output = []
	#Perform normalization on infile
	if type(args.infile) is file:
		#Input is stdin
			farsi_preprocess(file)
	else:
		#Input is list of files
		for f in args.infile:
			farsi_preprocess(f)
	#Write output to filename specified by user
	for line in output:
		print line
	return 0

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='''Preprocess Farsi text''')
	parser.add_argument('infile', nargs='*', type=argparse.FileType('r'), default=sys.stdin)
	args = parser.parse_args()
	#Run main function
	sys.exit(main(args))