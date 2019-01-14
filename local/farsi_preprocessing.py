#!/usr/bin/env python2
# -*- coding: UTF-8 -*-


# This only works in Python 2 because the farsiNorm script is written in Python 2
# This script takes a text file as input and performs the following 
# preprocessing steps for Farsi:
# 1. delete empty lines
# 2. split lines up so that there's only one sentence per line
# 3. get rid of punctuation
# 4. deduplication
# 5. use farsiNorm.py (gets rid of diacritics, ZWNJ, Arabic numerals and others)
# 6. normalize Arabic characters so that initial, middle, and end versions
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
		line = line.strip()
		# get rid of empty lines
		if line:
			# split lines that contain multiple sentences up
			line = line.split('. ')
			for l in line:
				text.append(l)
	#get rid of punctuation
	puncmarks = ['ARABIC COMMA', 'ARABIC QUESTION MARK', 'ARABIC SEMICOLON',\
	 'LEFT-POINTING DOUBLE ANGLE QUOTATION MARK', 'RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK']
	punc = punctuation
	for p in puncmarks:
	 	punc = punc + unicodedata.lookup(p)
	for sentence in text:
		s = ''.join(c for c in sentence.decode('utf-8') if c not in punc)
		nopunc.append(s.encode('utf-8'))
	# deduplication while at the same time keeping the order
	# (I first used set() for this and then couldn't find certain sentences while debugging
	# because they had been moved around due to set() not being ordered)
	#for sentence in OrderedDict.fromkeys(nopunc).keys():
	
##### moved deduplication down to the for i in out print loop
##### because the regex substitution came up with more duplicate sentences
	for sentence in nopunc:
		# normalization with farsiNorm.py
		normed.append(norm_farsi(sentence))
	for sentence in normed:
		# get rid of ZWNJ
		sentence = re.sub(u'\u200c'.encode('utf-8'), '', sentence)
		# substitute Arabic letters with Farsi equivalents
		# (the transcription data is somewhat 'polluted' with these two)
		sentence = re.sub('ك', 'ک', sentence)
		sentence = re.sub('ة', 'ه', sentence)
		# normalizes initial, mid, and end characters into
		# one universally recognised character
		new = unicodedata.normalize('NFKD', sentence.decode('utf-8'))
		out.append(new.encode('utf-8'))

	for i in OrderedDict.fromkeys(out).keys():
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