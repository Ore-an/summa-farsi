#!/usr/bin/env python3

import argparse, sys

# parser = argparse.ArgumentParser(description='''Create file called 'text' with transcripts''')
# parser.add_argument('corpusdir', nargs='+', default=sys.stdin)
# args = parser.parse_args()

# corpus = args.corpusdir[0]
# print(corpus)

for folder in ["all", "train", "test"]:
	utts = []
	fname = 'data/' + folder + '/wav.scp'
	with open(fname, encoding='utf-8', mode='r') as f:
		for line in f:
			line = line.strip().split('\t')
			utts.append((line[0], line[1]))

	fname = 'data/'+ folder + '/text'
	with open(fname, encoding='utf-8', mode='w') as o:
		for i in utts:
			file = i[1].strip('.wav') + '.txt'
			with open(file, encoding='windows-1256', mode='r') as f:
				text = ''
				for line in f:
					if line:
						#line = line.decode('windows-1256')
						text = text + line
				o.write(i[0])
				o.write('\t')
				o.write(text)
				o.write('\n')

print("Done creating transcript files")