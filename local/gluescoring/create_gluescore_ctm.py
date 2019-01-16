#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# uses the ctm.filt.filt (=argument 1) file from sclite scoring and "glues" the hypotheses together
# one line per scored reference file. outputs new file (=argument 2)

import sys

with open(sys.argv[1], encoding="utf8", mode="r") as f, open(sys.argv[2], encoding="utf8", mode="w") as o:
	fname = ''
	prev_fname = ''
	text = ''
	for line in f:
		line = line.split(' ')
		fname = line[0]
		if fname != prev_fname:
			if not prev_fname:
				text = text + fname + ' '
			else:
				text = text + '\n' + fname + ' '
		word = line[4]
		text = text + word + ' '
		prev_fname = fname
	o.write(text)