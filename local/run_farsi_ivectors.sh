#!/bin/bash

set -e -o pipefail

stage=0
nj=32
train_set=all_mfcc_hires
mix_set=mixed
test_set=summa_mfcc_hires
gmm=tri3b                # This specifies a GMM-dir from the features of the type you're training the system on;
                         # it should contain alignments for 'train_set'.

num_threads_ubm=32


. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp


for f in ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 2 ] && [ -f data/${train_set}_sp/feats.scp ]; then
  echo "$0: data/${train_set}_sp/feats.scp already exists."
  echo " ... Please either remove it, or rerun this script with stage > 2."
  exit 1
fi


if [ $stage -le 1 ]; then
  echo "$0: preparing directory for speed-perturbed data"
  cp -r data/all data/${train_set}
  for set in ${train_set} ${mix_set}; do
  utils/data/perturb_data_dir_speed_3way.sh data/$set data/${set}_sp
  done
fi

# use non-pitch features here because online iVector extraction isn't designed to deal with pitch
if [ $stage -le 2 ]; then
  cp -r data/summa data/${test_set}
	for datadir in ${train_set}_sp ${mix_set}_sp ${test_set}; do
	    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" data/${datadir}
	    steps/compute_cmvn_stats.sh data/${datadir}
	    utils/fix_data_dir.sh data/${datadir}
	done
fi

if [ $stage -le 3 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."

  mkdir -p exp/ivector/diag_ubm

  # train a diagonal UBM
  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" \
      --max-utts 10000 --subsample 2 \
       data/${train_set}_sp \
       exp/ivector/pca_transform

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    data/${train_set}_sp 512 \
    exp/ivector/pca_transform exp/ivector/diag_ubm
fi

if [ $stage -le 4 ]; then
  # Train the iVector extractor.  Use all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data.  The script defaults to an iVector dimension of
  # 100.
  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    data/${train_set}_sp exp/ivector/diag_ubm exp/ivector/extractor || exit 1;
fi

if [ $stage -le 5 ]; then
	#extract ivectors for train set
  for set in ${train_set} ${mix_set}; do
    ivectordir=exp/ivector/ivectors_${set}_sp
	  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj data/${set}_sp exp/ivector/extractor $ivectordir
  done

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for data in ${test_set}; do
    nspk=$(wc -l <data/${data}/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "${nspk}" \
      data/${data} exp/ivector/extractor \
      exp/ivector/ivectors_${data}
  done
fi