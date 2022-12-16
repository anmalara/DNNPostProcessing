#!/bin/bash

set -x

echo "args: $@"

filepath=$1
model=$2
data=$3
flag_test=$4
train=$5
val=$6
test=$7
n_gpus=$8
n_epochs=$9
extra_name=${10}

WORKDIR=$(pwd)

# Download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda_install.sh
bash miniconda_install.sh -b -p ${WORKDIR}/miniconda
export PATH=$WORKDIR/miniconda/bin:$PATH
source $WORKDIR/miniconda/etc/profile.d/conda.sh
conda create -n weaver python=3.7 --yes
conda activate weaver
# # to be used in case one would want to change storage path
# conda create python=3.7 --yes -p $WORKDIR/miniconda/envs/weaver
# conda activate $WORKDIR/miniconda/envs/weaver
# install weaver and clone particle_transformer
pip install numpy pandas scikit-learn scipy matplotlib tqdm PyYAML
pip install uproot3 awkward0 lz4 xxhash
pip install tables
pip install onnxruntime-gpu
pip install tensorboard
pip install torch==1.12.1
pip install pyarrow
pip install weaver-core
git clone https://github.com/LEAF-HQ/DNNPostProcessing.git

cd DNNPostProcessing
# # CUDA environment setup
# export PATH=$PATH:/usr/local/cuda-10.2/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
# export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/lib64

echo ${filepath}/train_VBF.sh -m ${model} -d ${data} -f ${flag_test} -t "${train}" -v "${val}" -x "${test}" -g ${n_gpus} -e ${n_epochs} -n ${extra_name}
     ${filepath}/train_VBF.sh -m ${model} -d ${data} -f ${flag_test} -t "${train}" -v "${val}" -x "${test}" -g ${n_gpus} -e ${n_epochs} -n ${extra_name}

[ -d "runs/" ] && tar -caf output.tar trainings/${model}/ runs/ || tar -caf output.tar trainings/${model}/
