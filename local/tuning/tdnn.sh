#!/bin/bash

## loosely based on a script from babel egs

#Results on Persian SUMMA TestSet v1.1: 59.5% WER
# with speed perturbed data: 59.1% WER

set -e

stage=0
affix=
train_stage=-10
common_egs_dir=
reporting_email=
remove_egs=true
feature=mfcc_hires_pitch
cmvn=false
num_epochs=2

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ -z $feature ]; then
    echo "You have to define a feature type, e.g. fbank23, fbank23_pitch, or mfcc_hires"
fi

if [ "$cmvn" == "false" ] || [ "$cmvn" == "False" ]; then
	suffix="_no_cmvn"
else
	suffix=""
fi

if [ "$num_epochs" -ne 2 ]; then
	suffix=${suffix}_ep${num_epochs}
fi

dir=exp/tdnn_${feature}${suffix}${affix}
ali_dir=exp/tri3b_ali${affix}
lang=data/lang_v3.3gm.p07
decode_suff="" # for different LM
feats_dim=`feat-to-dim scp:data/all_${feature}${affix}/feats.scp -`
echo "THIS IS FEATS_DIM"
echo $feats_dim

export CUDA_VISIBLE_DEVICES=2,3

if [ $stage -le 9 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $ali_dir/tree | grep num-pdfs | awk '{print $2}')

  mkdir -p $dir/configs
    cat <<EOF > $dir/configs/network.xconfig
  input dim=$feats_dim name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=512
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=512
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=512
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn6 input=Append(-6,-3,0) dim=512

  output-layer name=output input=tdnn6 dim=$num_targets max-change=1.5
EOF

    steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi



if [ $stage -le 10 ]; then

  steps/nnet3/train_dnn.py --stage=$train_stage \
			   --cmd="$decode_cmd" \
			   --feat.cmvn-opts="--norm-means=$cmvn --norm-vars=$cmvn" \
			   --trainer.num-epochs $num_epochs \
			   --trainer.optimization.num-jobs-initial 2 \
			   --trainer.optimization.num-jobs-final 2 \
			   --trainer.optimization.initial-effective-lrate 0.004 \
			   --trainer.optimization.final-effective-lrate 0.00017 \
			   --egs.dir "$common_egs_dir" \
			   --cleanup.remove-egs $remove_egs \
			   --cleanup.preserve-model-interval 100 \
			   --use-gpu true \
			   --feat-dir=data/all_${feature}${affix} \
			   --ali-dir $ali_dir \
			   --lang ${lang} \
			   --dir=$dir  || exit 1;

fi


if [ $stage -le 11 ]; then
	utils/mkgraph.sh $lang $dir $dir/graph_v3.3gm.p07
    steps/nnet3/decode.sh --nj 12 --cmd "$decode_cmd" --skip_scoring true --skip_diagnostics true \
	 $dir/graph_v3.3gm.p07 data/summa_${feature} $dir/decode_summa_v3.3gm.p07
fi

if [ $stage -le 12 ]; then
    run.pl LMWT=8:12 $dir/decode_summa_v3.3gm.p07/scoring/log/best_path.LMWT.log local/get_ctm.sh --stm data/summa/stm --glm data/summa/glm LMWT 0.0 $lang data/summa_${feature} $dir/final.mdl $dir/decode_summa_v3.3gm.p07
fi
grep Sum $dir/decode_summa_v3.3gm.p07/score_*/*/*.sys

wait;
exit 0;
