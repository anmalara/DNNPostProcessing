#!/bin/bash

set -x

export DATADIR_VBF='/eos/user/h/hevard/datasets/test'

echo "args: $@"

# set the dataset dir via `DATADIR_QuarkGluon`
DATADIR=${DATADIR_VBF}
[[ -z $DATADIR ]] && DATADIR='./datasets/QuarkGluon'

# set a comment via `COMMENT`
suffix=${COMMENT}

# PN, PFN, PCNN, ParT
modelopts="networks/VBFmlp.py"
lr="1e-3"
extraopts=""

weaver \
    --data-train "${DATADIR}/MC__GluGluHToZZTo4L_M130_standard_UL18_[1-3].root" \
    --data-test "${DATADIR}/MC__GluGluHToZZTo4L_M130_standard_UL18_4.root" \
    --data-config data/VBF/VBFmlp.yaml --network-config $modelopts \
    --model-prefix training/VBF/{auto}${suffix}/net \
    --num-workers 1 --fetch-step 1 --in-memory --train-val-split 0.8889 \
    --batch-size 512 --samples-per-epoch 16000 --samples-per-epoch-val 2000 --num-epochs 2 \
    --start-lr $lr --optimizer ranger --log logs/VBF_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard VBF_${suffix} \
    ${extraopts} "${@:3}"
#    --batch-size 512 --samples-per-epoch 16000 --samples-per-epoch-val 2000 --num-epochs 2 --gpus 0 \
