#!/bin/bash

# In hindsight, this recipe might not have been optimised properly in terms of layer size and dimensions
#Results on Persian SUMMA TestSet v1.1 (with speed_perturb=true): 60.1% WER

set -e

feature=mfcc_hires_pitch
train_set=all_${feature}

# configs for 'chain'
affix=
stage=0
train_stage=-10
get_egs_stage=-10
speed_perturb=true
decode_iter=
cmvn=false
fss=3 #frame-subsampling-factor (best kept at default which is 3)
num_leaves=5000 # 7000 was the default for this recipe

# training options
num_epochs=4
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

export CUDA_VISIBLE_DEVICES=0,1

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

train_set=${train_set}${suffix}
dir=exp/chain${cmvnsuffix}${suffix}
dir=${dir}${affix:+_$affix}
ali_dir=exp/tdnn_ali${suffix}
treedir=exp/chain_tree
lang=data/lang_chain

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/${train_set} \
    data/lang_v3.3gm.p07 exp/tri3b exp/tri3b_lats$suffix
  rm exp/tri3b_lats$suffix/fsts.*.gz # save space
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
      --cmd "$train_cmd" $num_leaves data/${train_set} $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  feats_dim=`feat-to-dim scp:data/${train_set}/feats.scp -`
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=$feats_dim name=input

  relu-renorm-layer name=tdnn1 dim=625
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=625
  relu-renorm-layer name=tdnn4 input=Append(-3,0,3) dim=625
  relu-renorm-layer name=tdnn5 input=Append(-3,0,3) dim=625
  relu-renorm-layer name=tdnn6 input=Append(-3,0,3) dim=625
  relu-renorm-layer name=tdnn7 input=Append(-3,0,3) dim=625
  ## adding the layers for chain branch
  relu-renorm-layer name=prefinal-chain input=tdnn7 dim=625 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  relu-renorm-layer name=prefinal-xent input=tdnn7 dim=625 target-rms=0.5
  relu-renorm-layer name=prefinal-lowrank-xent input=prefinal-xent dim=64 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts "--norm-means=$cmvn --norm-vars=$cmvn" \
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
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --lat-dir exp/tri3b_lats$suffix \
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

      run.pl LMWT=8:12 $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff}/scoring/log/best_path.LMWT.log local/get_ctm.sh --stm data/summa/stm --glm data/summa/glm LMWT 0.0 $lang data/summa_${feature} $dir/final.mdl $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff}
      grep Sum $dir/decode_*/score_*/*/*.sys
  done
fi

wait;
exit 0;
