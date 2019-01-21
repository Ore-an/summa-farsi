#!/bin/bash
# Copyright 2019 Simon Vandieken
[ -f cmd.sh ] && source ./cmd.sh || echo "cmd.sh not found. Jobs may not execute properly."
# all the commands are run.pl for use on starariel

set -e

stage=0
nj=32
decode_nj=12
. path.sh || { echo "Cannot source path.sh"; exit 1; }
. ./utils/parse_options.sh

yourUUN=svandiek
# this is Bagher's clean data set; the data prep scripts further down the line make it into a training folder at data/all_$feature
# The scripts also separate it into a training and a test set which were used before the SUMMA TestSet was available. Please ignore them.
export CORPUS=/group/project/summa/PersianNewsDBBagher/PersianNewsDB/new
# additionally this recipe assumes you have two other data sets at hand:
# 1) 190h of aligned Farsi Euronews data which can be found on starariel:/disk/scratch1/svandiek/euronews_persian/data/aligned_uttfix
# This data set isn't as clean as Bagher's data set which is why we mix those two datasets into one at a later stage.
# Our experiments have shown that first training on the mixed data set and then finetuning on the clean one gives the best results.

scp -r $yourUUN@starariel:/disk/scratch1/svandiek/euronews_persian/data/aligned_uttfix data/

# 2) The SUMMA TestSet, which to the best of our knowledge is not publicly available (as are the two other data sets)
# All of our results were done using v1.1 with some minor tweaks to it to get it to work with Kaldi
# Some peculiarities about the TestSet:
# - Segmentation does not seem to be super accurate as shown by the fact that "gluescoring" our best results
# 	gives nearly 10% absolute WER improvement.
# - It does not account for music, general noise or overlapping speech.
# - One of the twelve files does not contain any audio past the 39 minute mark. Transcriptions are also not available for that part, so this does not affect scoring.

scp -r $yourUUN@starariel:/disk/scratch1/svandiek/persian/data/summa data/

# Our results are reported using starariel:/disk/scratch1/svandiek/persian/data/summa

# All our nnet3 experiments do not use CMVN by default. You can get better results offline when using CMVN but due to the nature of the SUMMA platform
# as an online decoding platform we would have to use global CMVN which has given worse results in our experience.

wget http://data.cstr.ed.ac.uk/summa/release/farsi_asr_lm.3gm.150k.p07.gz
wget http://data.cstr.ed.ac.uk/summa/release/farsi_asr_lm.4gm.full.gz
wget http://data.cstr.ed.ac.uk/summa/release/farsi_asr_grapheme_dict.txt

LM=farsi_asr_lm.3gm.150k.p07.gz
lm_name=v3.3gm.p07

num_states=3100
num_gauss=50000

feature=mfcc_hires_pitch

if [ $stage -le 0 ]; then
####### data prep
	local/farsi_data_prep.sh $CORPUS

	###### dict prep
	local/farsi_dict_prep.sh

	utils/prepare_lang.sh --position-dependent-phones true data/local/dict [EH] data/local/lang data/lang

	###### LM prep
	utils/format_lm.sh data/lang $LM data/local/dict/lexicon.txt data/lang_${lm_name}
fi

if [ $stage -le 1 ]; then
	# Make MFCC features
	utils/combine_data.sh data/mixed data/all data/aligned_uttfix
	cp -r data/all data/all_$feature
	cp -r data/summa data/summa_${feature}
	# we don't need to create the features for data/mixed yet because we're not going to use them yet.
	for set in all_${feature} summa_${feature}; do
		steps/make_mfcc_pitch.sh --nj $nj --cmd "$train_cmd" --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf data/$set
		steps/compute_cmvn_stats.sh data/$set
		wait;
	done
fi

if [ $stage -le 2 ]; then
	# monophones
	mkdir -p exp/mono
	steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/all_${feature} data/lang exp/mono >& exp/mono/train.log

	graph_dir=exp/mono/graph_${lm_name}
	mkdir -p $graph_dir
	utils/mkgraph.sh data/lang_${lm_name} exp/mono $graph_dir

	steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" $graph_dir data/summa_${feature} exp/mono/decode_${lm_name}
fi

if [ $stage -le 3 ]; then
	# triphones
	mkdir -p exp/mono_ali
	steps/align_si.sh --nj $nj --cmd "$train_cmd" \
	data/all_${feature} data/lang exp/mono exp/mono_ali \
	>& exp/mono_ali/align.log


	mkdir -p exp/tri1
	steps/train_deltas.sh --cmd "$train_cmd" \
	--cluster-thresh 100 $num_states $num_gauss data/all_${feature} data/lang \
	exp/mono_ali exp/tri1 >& exp/tri1/train.log
fi

if [ $stage -le 4 ]; then
	# Decode tri1

	graph_dir=exp/tri1/graph_${lm_name}
	mkdir -p $graph_dir
	utils/mkgraph.sh data/lang_${lm_name} exp/tri1 $graph_dir

	steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" $graph_dir data/summa_${feature} exp/tri1/decode_${lm_name}
fi

if [ $stage -le 5 ]; then
	# Train tri2a, which is deltas + delta-deltas
	mkdir -p exp/tri1_ali
	steps/align_si.sh --nj $nj --cmd "$train_cmd" data/all_${feature} data/lang exp/tri1 exp/tri1_ali >& exp/tri1_ali/tri1_ali.log

	mkdir -p exp/tri2a
	steps/train_deltas.sh --cmd "$train_cmd" --cluster-thresh 100 $num_states $num_gauss data/all_${feature} data/lang exp/tri1_ali exp/tri2a >& exp/tri2a/train.log
	wait;

	# Decode tri2a
	graph_dir=exp/tri2a/graph_${lm_name}
	mkdir -p $graph_dir
	utils/mkgraph.sh data/lang_${lm_name} exp/tri2a $graph_dir

	steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" $graph_dir data/summa_${feature} exp/tri2a/decode_${lm_name}
fi

if [ $stage -le 6 ]; then
	# Train tri2b, which is LDA+MLLT    
	mkdir -p exp/tri2b
	steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" $num_states $num_gauss data/all_${feature} \
		data/lang exp/tri1_ali exp/tri2b >& exp/tri2b/tri2_ali.log 
	wait;


	# Decode tri2b
	graph_dir=exp/tri2b/graph_${lm_name}
	mkdir -p $graph_dir
	utils/mkgraph.sh data/lang_${lm_name} exp/tri2b $graph_dir  

	steps/decode.sh --nj $decode_nj --cmd "$decode_cmd" $graph_dir data/summa_${feature} exp/tri2b/decode_${lm_name} 
fi

if [ $stage -le 7 ]; then

	# Train tri3b, which is LDA+MLLT+SAT.

	mkdir -p exp/tri2b_ali
	steps/align_si.sh --nj $nj --cmd "$train_cmd" --use-graphs true data/all_${feature} data/lang exp/tri2b exp/tri2b_ali >& exp/tri2b_ali/align.log

	mkdir -p exp/tri3b
	steps/train_sat.sh --cmd "$train_cmd" --cluster-thresh 100 $num_states $num_gauss data/all_${feature} data/lang exp/tri2b_ali exp/tri3b >& exp/tri3b/train.log
	wait;

	# Decode 3b

	graph_dir=exp/tri3b/graph_${lm_name}
	mkdir -p $graph_dir
	utils/mkgraph.sh data/lang_${lm_name} exp/tri3b $graph_dir

	mkdir -p exp/tri3b/decode_${lm_name}
	steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" $graph_dir data/summa_${feature} exp/tri3b/decode_${lm_name}

fi

if [ $stage -le 8 ]; then
	# align tri3b
	mkdir -p exp/tri3b_ali
	steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" data/all_${feature} data/lang exp/tri3b exp/tri3b_ali
fi

if [ $stage -le 9 ]; then
	# trains ivector extractor and extracts ivectors for the training, the mixed, and the test set
	# this can arguably be run at a later stage
	local/run_farsi_ivectors.sh
fi

if [ $stage -le 10 ]; then
	# create speed perturbed sets
	#rm data/mixed_sp/feats.scp
	for set in data/all_${feature} data/mixed; do
		utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true $set ${set}_sp
		steps/make_mfcc_pitch.sh --nj $nj --cmd "$train_cmd" --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf ${set}_sp
		steps/compute_cmvn_stats.sh ${set}_sp
	done
fi

#add WERs to all the local/tuning scripts

#TDNN CE
local/tuning/tdnn.sh --feature $feature

#+SP
# align speed perturbed data with tri3b
mkdir -p exp/tri3b_ali_sp
steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" data/all_${feature}_sp data/lang exp/tri3b exp/tri3b_ali_sp

local/tuning/tdnn.sh --feature $feature --affix _sp

# get alignments from TDNN because they should be better than the previous ones
steps/nnet3/align.sh --nj $nj --use-gpu false data/all_${feature}_sp data/lang exp/tdnn_${feature}_no_cmvn_sp exp/tdnn_ali_sp

#+SPchain
local/tuning/tdnn_chain.sh --feature $feature --speed_perturb true
	
#TDNN-F +LF-MMI+SP
# start at --stage 10 because the lattice alignments are already there if you ran the previous script
local/tuning/tdnnf_chain.sh --stage 10

mkdir -p exp/tdnn_mixed_ali_sp
steps/nnet3/align.sh --nj $nj --use-gpu false data/mixed_sp data/lang exp/tdnn_${feature}_no_cmvn_sp exp/tdnn_mixed_ali_sp

#+euronews data mixed in
# this will nonetheless use data/mixed_sp
local/tuning/tdnnf_chain.sh --train_set data/mixed

#+mix+ivec
local/tuning/tdnnf_chain_ivec.sh

#+mix+ivec+finetune
local/tuning/tdnnf_chain_finetune.sh --feature $feature

#+4gm rescore
utils/format_lm.sh data/lang farsi_asr_lm.4gm.full.gz data/local/dict/lexicon.txt data/lang_v3.4gm
# NB this uses about 90 GB of memory!
local/lmrescore_summa.sh --self-loop-scale 1.0 data/lang_${lm_name} data/lang_v3.4gm data/summa \
	exp/chain_tdnnf_ivec_mix_+finetune_no_cmvn_sp/decode_summa_${feature}_chain \
	exp/chain_tdnnf_ivec_mix_+finetune_no_cmvn_sp/decode_summa_4gmrescore 

#+gluescore
python3 local/gluescoring/create_gluescore_ctm.py \ 
	exp/chain_tdnnf_ivec_mix_+finetune_no_cmvn_sp/decode_summa_4gmrescore/score_10/penalty_0.0/summa.ctm.filt.filt \
	exp/chain_tdnnf_ivec_mix_+finetune_no_cmvn_sp/decode_summa_4gmrescore/score_10/penalty_0.0/gluescore_ctm.filt.filt

local/gluescoring/glue_stm_before.sh

compute-wer ark:data/summa/reference_before_plural_fix ark:exp/chain_tdnnf_ivec_mix_+finetune_no_cmvn_sp/decode_summa_4gmrescore/score_10/penalty_0.0/gluescore_ctm.filt.filt

#+plural fix
local/gluescoring/glue_stm_after.sh

compute-wer ark:data/summa/reference_after_plural_fix ark:exp/chain_tdnnf_ivec_mix_+finetune_no_cmvn_sp/decode_summa_4gmrescore/score_10/penalty_0.0/gluescore_ctm.filt.filt
