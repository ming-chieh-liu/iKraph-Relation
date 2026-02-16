#! /bin/bash

# Get type from first argument, default to "_curated" if not provided
type="${1:-_litcoin_400_curated}"
# Get GPU from second argument, default to 1 if not provided
gpu="${2:-1}"
echo "Postfix is '${type}'"
echo "Using GPU: ${gpu}"

export CUDA_VISIBLE_DEVICES=${gpu}

python generate_train_configs.py --posfix ${type}

if true; then
sfs=(0.02) #(0.0 0.02)
lrs=(3e-05) #(3e-05)
bss=(16) #(16 32)
fidxs=(0 1 2 3 4)
models=("roberta" "pmbert")
for model in ${models[@]}
do
for sf in ${sfs[@]}
do
for lr in ${lrs[@]}
do
for bs in ${bss[@]}
do
for fidx in ${fidxs[@]}
do
if [ "$model" == "pmbert" ]; then
echo "Running PMBERT with sf=${sf}, fidx=${fidx}, bs=${bs}, lr=${lr}"
python -u ./run_modeling_bert.py ./configs${type}/config_pmbert_ls${sf}_split_${fidx}_${bs}_${lr}.json
elif [ "$model" == "roberta" ]; then
echo "Running RoBERTa with sf=${sf}, fidx=${fidx}, bs=${bs}, lr=${lr}"
python -u ./run_modeling_roberta.py ./configs${type}/config_roberta_ls${sf}_split_${fidx}_${bs}_${lr}.json
fi
done 
done
done
done
done
fi

