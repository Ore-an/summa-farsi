#!/usr/bin/env python3

# script that helps create wav.scp, utt2spk, spk2utt, and text files
# this script should be called by farsi_data_prep.sh
# input arg should be location of CORPUS

import re
import sys
from collections import defaultdict
from collections import namedtuple

trans = []
with open('transcripts_paths', encoding='utf-8', mode='r') as f:
	for line in f:
		line = line.strip()
		trans.append(line)

audio = []
with open('audio_paths', encoding='utf-8', mode='r') as f:
	for line in f:
		line = line.strip()
		audio.append(line)


# filter out utts with either missing transcript or missing audio files
pairs = []

for t in trans:
	a = t.strip('.txt') + '.wav'
	if a in audio:
		pairs.append((t, a))

utterance = namedtuple('utt', 'uttID, spkID, wavfile')

utts = []
for p in pairs:
	utt = p[0].strip('.txt').split('/')[-1]
	spk = utt[:-3]
	# it would be possible to include transcripts
	utts.append(utterance(uttID=utt, spkID=spk, wavfile=p[1]))

spk2utt = defaultdict(list)
utt2spk = defaultdict(str)

for utt in utts:
	spk2utt[utt.spkID].append(utt.uttID)
	utt2spk[utt.uttID] = utt.spkID

with open('data/all/spk2utt', encoding='utf-8', mode='w') as o:
	for speaker in sorted(spk2utt.keys()):
		o.write(speaker)
		o.write('\t')
		for utt in spk2utt[speaker]:
			o.write(utt)
			o.write(' ')
		o.write('\n')

with open('data/all/utt2spk', encoding='utf-8', mode='w') as o:
	for utt in sorted(utt2spk.keys()):
		o.write(utt)
		o.write('\t')
		o.write(utt2spk[utt])
		o.write('\n')

with open('data/all/wav.scp', encoding='utf-8', mode='w') as o:
	for utt in utts:	
		o.write(utt.uttID)
		o.write('\t')
		o.write(utt.wavfile)
		o.write('\n')

print("Done creating spk2utt, utt2spk, wav.scp")