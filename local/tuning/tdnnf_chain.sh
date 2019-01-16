#!/bin/bash

# 
set -e

#Results on Persian SUMMA TestSet v1.1 with clean data set: 52.8% WER
# with mixed data set: % WER

train_set=data/all_mfcc_hires_pitch

# configs for 'chain'
feature=mfcc_hires_pitch
ivec_feature=mfcc_hires
affix=
stage=0
train_stage=-10
get_egs_stage=-10
speed_perturb=true
decode_iter=
cmvn=false
fss=3 #frame-subsampling-factor

# backstitch options
alpha=0.3
back_interval=1
num_epochs=8 # num_epochs needs to be twice as high according to https://github.com/kaldi-asr/kaldi/issues/1942

# training options
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=2
num_jobs_final=2
minibatch_size=128
frames_per_eg=150
remove_egs=false
common_egs_dir=
xent_regularize=0.1

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

export CUDA_VISIBLE_DEVICES=2,3

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ "$cmvn" == "false" ] || [ "$cmvn" == "False" ]; then
  cmvnsuffix="_no_cmvn"
else
  cmvnsuffix=""
fi

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

mix=
if [ "$train_set" == "data/mixed_sp" ] || [ "$train_set" == "data/mixed" ]; then
  mix="_mixed"
fi

train_set=${train_set}$suffix
dir=exp/chain_tdnnf${mix}${cmvnsuffix}${suffix}
dir=${dir}${affix:+_$affix}
ali_dir=exp/tdnn${mix}_ali${suffix}   # exp/v3/tri3_ali${suffix}
treedir=exp/chain_tree
lang=data/lang_chain


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri3b_ali/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${train_set} \
    data/lang_v3.3gm.p07 exp/tri3b exp/tri3b${mix}_lats$suffix
  rm exp/tri3b${mix}_lats$suffix/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang_v3.3gm.p07 $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor $fss \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 $train_set $lang $ali_dir $treedir
fi

affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim-continuous=true"
tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66"

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  feats_dim=`feat-to-dim scp:${train_set}/feats.scp -`
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig

  input dim=$feats_dim name=input

  # please note that it is important to have input layer with the name=input

  relu-renorm-layer name=tdnn1 $affine_opts dim=512
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=512 bottleneck-dim=128 time-stride=3

  ## adding the layers for chain branch
  relu-renorm-layer name=prefinal-chain input=tdnnf12 dim=512 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  relu-renorm-layer name=prefinal-xent input=tdnnf12 dim=512 target-rms=0.5
  relu-renorm-layer name=prefinal-lowrank-xent input=prefinal-xent dim=64 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts "--norm-means=$cmvn --norm-vars=$cmvn" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --chain.frame-subsampling-factor $fss \
    --chain.alignment-subsampling-factor $fss \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.backstitch-training-scale $alpha \
    --trainer.optimization.backstitch-training-interval $back_interval \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.optimization.proportional-shrink 20 \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir ${train_set} \
    --tree-dir $treedir \
    --lat-dir exp/tri3b${mix}_lats$suffix \
    --dir $dir  || exit 1;

fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_v3.3gm.p07/ $dir $dir/graph_chain
fi

decode_suff=chain
graph_dir=$dir/graph_chain
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in summa_${feature}; do
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 12 --cmd "$decode_cmd" $iter_opts --skip_scoring true --skip_diagnostics true \
          $graph_dir data/${decode_set} $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff};

     	run.pl LMWT=8:12 $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff}/log/best_path.LMWT.log local/get_ctm.sh --stm data/summa/stm --glm data/summa/glm LMWT 0.0 $lang data/summa_$feature $dir/final.mdl $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff}
    	grep Sum $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff}/score_*/*/*.sys
  done
fi

wait;
exit 0;
