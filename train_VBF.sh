#!/bin/bash

# Example of usage
#./train_VBF.sh
#./train_VBF.sh -m mlp_pf -d VBF_features
#./train_VBF.sh -m deepak8_pf -d VBF_features
#./train_VBF.sh -m particlenet_pf -d VBF_points_features
#./train_VBF.sh -e "--batch-size 512 --num-n_epochs 3"
#./train_VBF.sh -p pred_all -o trainings/particlenet_pf/*VBF_points_features_epoch_10_cat012

set -x

echo "args: $@"

# Defining nominal variables
model="particlenet_pf"
data="VBF_points_features"
flag_test=""
n_gpus=""
n_epochs="1"
extra_name=""
inputdir="/eos/home-a/anmalara/Public/DNNInputs"
data_train_ref="${inputdir}/eventCategory_0/MC*M1[2][4-6]*UL1[6-7-8]*.root"
data_val_ref="${inputdir}/eventCategory_0/MC*M1[2][0]*UL1[6-7-8]*.root"
data_test_ref="${inputdir}/eventCategory_0/MC*M1[3][0]*UL1[6-7-8]*.root"

data_train="${data_train_ref}"
data_val="${data_val_ref}"
data_test="${data_test_ref}"
mode="train"
outputdir=""


while getopts m:d:f:t:v:x:g:e:n:p:o: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        d) data=${OPTARG};;
        f) flag_test=${OPTARG};;
        t) data_train=${OPTARG};;
        v) data_val=${OPTARG};;
        x) data_test=${OPTARG};;
        g) n_gpus=${OPTARG};;
        e) n_epochs=${OPTARG};;
        n) extra_name=${OPTARG};;
        p) mode=${OPTARG};;
        o) outputdir=${OPTARG};;
    esac
done

if [[ ${extra_name} != "" ]]; then
    extra_name="_"${extra_name}
fi


if [[ ${outputdir} == "" ]]; then
    outputdir="trainings/${model}/{auto}_${data}${extra_name}"
fi

output_name="pred.root"
model_config="models/${model}.py"
data_config="data/${data}.yaml"

train_opts="--num-epochs "${n_epochs}
batch_opts="--batch-size 512 --start-lr 1e-3"

if [[ $flag_test == "test" ]]; then
    data_train="${data_train_ref}"
    data_val="${data_val_ref}"
    data_test="${data_test_ref}"
fi

if [[ ${n_gpus} == "0" ]]; then
    n_gpus=""
elif [[ ${n_gpus} == "1" ]]; then
    n_gpus="0"
elif [[ ${n_gpus} == "2" ]]; then
    n_gpus="0,1"
elif [[ ${n_gpus} == "3" ]]; then
    n_gpus="0,1,2"
elif [[ ${n_gpus} == "4" ]]; then
    n_gpus="0,1,2"
fi

if [[ ${mode} == "train" ]]; then
    weaver \
        --data-train "${data_train}" --data-val "${data_val}" --data-test "${data_test}" \
        --data-config ${data_config} --network-config ${model_config} \
        --model-prefix "${outputdir}/net" --log "${outputdir}/log.log" \
        ${train_opts} ${batch_opts} --gpus "${n_gpus}" \
        --num-workers 4 --fetch-step 0.01 \
        --optimizer ranger --predict-output ${output_name}\
        --tensorboard "${model}_${data}${extra_name}"
elif [[ ${mode} == *"pred"* ]]; then
    n_gpus="0"
    nvidia-smi || n_gpus=""
    data_train="${inputdir}/eventCategory_[0-2]/MC*M1[2][4-6]*UL1[6-7-8]*.root"
    data_val="${inputdir}/eventCategory_[0-2]/MC*M1[2][0]*UL1[6-7-8]*.root"
    data_test="${inputdir}/eventCategory_[0-2]/MC*M1[3][0]*UL1[6-7-8]*.root"
    output_name="${mode}_test.root"
    weaver \
        --predict --data-test "${data_test}" \
        --data-config ${data_config} --network-config ${model_config} \
        --model-prefix "${outputdir}/net" --log "${outputdir}/log_predict.log" \
        ${batch_opts} --gpus "${n_gpus}" --num-workers 4 --fetch-step 0.0 \
        --predict-output ${output_name}
    output_name="${mode}_train.root"
    weaver \
        --predict --data-test "${data_train}" \
        --data-config ${data_config} --network-config ${model_config} \
        --model-prefix "${outputdir}/net" --log "${outputdir}/log_predict.log" \
        ${batch_opts} --gpus "${n_gpus}" --num-workers 4 --fetch-step 0.0 \
        --predict-output ${output_name}
    output_name="${mode}_val.root"
    weaver \
        --predict --data-test "${data_val}" \
        --data-config ${data_config} --network-config ${model_config} \
        --model-prefix "${outputdir}/net" --log "${outputdir}/log_predict.log" \
        ${batch_opts} --gpus "${n_gpus}" --num-workers 4 --fetch-step 0.0 \
        --predict-output ${output_name}
fi

#--samples-per-epoch 100000 --samples-per-epoch-val 20000
#--tensorboard VBF_${suffix} \
#--in-memory --train-val-split 0.8889
# --num-workers 1 --fetch-by-files --fetch-step 10\
