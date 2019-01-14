#!/usr/bin/env bash


# script to take data from /group/project/summa/PersianNewsDBBagher/PersianNewsDB and create the following files in $PWD (which should be $KALDI_ROOT/persian)
#data/all/wav.scp - mapping from utt ID to wav file
#data/all/text - should be preprocessed afterwards!
#data/all/utt2spk
#data/all/spk2utt

. path.sh || { echo "Cannot source path.sh"; exit 1; }

. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 [opts] <corpus>"
  echo "please indicate where the corpus is located"
  exit 1;
fi

CORPUS=$1

find $CORPUS | grep .txt > transcripts

for file in `cat transcripts`;do
	realpath $file >> transcripts_paths
done

find $CORPUS | grep .wav > audio

for file in `cat audio`;do
	realpath $file >> audio_paths
done

if [ ! -d data/all ];then
	mkdir -p data/all
fi
# filter out utts with either transcription or audio file missing

python3 local/create_data_prep_files.py 

rm audio
rm audio_paths
rm transcripts
rm transcripts_paths

# split data into two sets
for folder in all train test;do
	if [ ! -d data/$folder ];then
		mkdir data/$folder
	fi
done

${KALDI_ROOT}/egs/wsj/s5/utils/nnet/subset_data_tr_cv.sh data/all data/train data/test || { echo "Error splitting data."; exit 1; }


# text files are now missing
python3 local/create_transcript_file.py $CORPUS

# Preprocessing transcripts
# gets rid of unwanted characters, punctuation, etc.
for folder in all train test; do
	mkdir data/$folder/tmp
	cut -f1 data/$folder/text > data/$folder/tmp/uttids
	cut -f2 data/$folder/text > data/$folder/tmp/trans
	python2 local/farsi_preprocess_transcripts.py data/$folder/tmp/trans > data/$folder/tmp/trans_prep
	# create backup of old text file in case preprocessing goes horribly wrong
	mv data/$folder/text data/$folder/text_backup
	# insert assertion like wc -l uttids and wc -l trans_prep have to be the same
	a=$(wc -l data/$folder/tmp/uttids | cut -d' ' -f1)
	b=$(wc -l data/$folder/tmp/trans_prep | cut -d' ' -f1)
	if [ "$a" -ne "$b" ]; then
		echo "Number of utterance has decreased during preprocessing. Something went wrong.";
		exit 1
	fi
	paste data/$folder/tmp/uttids data/$folder/tmp/trans_prep > data/$folder/text
	rm -r data/$folder/tmp
done

echo "Done with data preparation"