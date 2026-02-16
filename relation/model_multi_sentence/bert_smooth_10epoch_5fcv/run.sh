#!/bin/bash -e

# Get type from first argument, default to "_curated" if not provided
type="${1:-_litcoin_400}"
transform_method="${2:-entity_mask}"
# Get GPU from third argument, default to 1 if not provided
gpu="${3:-1}"
echo "Postfix is '${type}'"
echo "Using GPU: ${gpu}"

export CUDA_VISIBLE_DEVICES=${gpu}

python generate_train_configs.py --posfix ${type} --transform_method ${transform_method}

if true; then
sfs=(0.02) #(0.0 0.02)
lrs=(3e-05) #(3e-05)
bss=(16) #(16 32)
fidxs=(0 1 2 3 4) # (0 1 2 3 4)
models=("roberta")

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
echo "Running RoBERTa with sf=${sf}, fidx=${fidx}, bs=${bs}, lr=${lr}"
python -u ./modeling.py ./configs${type}_${transform_method}/config_roberta_ls${sf}_split_${fidx}_${bs}_${lr}.json
done 
done
done
done
done
fi

# pretrain="../pre_trained_models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf"
# export HF_HOME=cache
# for rs in 1 2 3 4 6 7 8 9 10 12 14 15 40 42;
# do
# echo "roberta_BS16_lr3e-5_RS${rs}" 
# python -u modeling.py --batch_size=16 \
#                   --RD ${rs} \
#                   --train_set train.json \
#                   --valid_set devel.json \
#                   --test_set  test.json \
#                   --pretrain ${pretrain} \
# 	              --epochs 20 \
#                   --output_path "roberta_BS16_lr3e-5_RS${rs}" \
# 2>&1 |tee log.roberta_BS16_lr3e-5_RS${rs} 
done
