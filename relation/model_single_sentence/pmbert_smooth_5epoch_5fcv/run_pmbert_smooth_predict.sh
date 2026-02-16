#! /bin/bash

# Usage: ./run_pmbert_smooth_predict.sh <pred_configs_dir>
# Example: ./run_pmbert_smooth_predict.sh ./pred_configs

PRED_CONFIGS_DIR=${1:-"./pred_configs"}
gpu="${2:-1}"
echo "Config path is '${PRED_CONFIGS_DIR}'"
echo "Using GPU: ${gpu}"

export CUDA_VISIBLE_DEVICES=${gpu}

if [ ! -d "$PRED_CONFIGS_DIR" ]; then
    echo "Error: Directory $PRED_CONFIGS_DIR does not exist"
    exit 1
fi

ckplist=(`ls ${PRED_CONFIGS_DIR}/*.json 2>/dev/null`)

if [ ${#ckplist[@]} -eq 0 ]; then
    echo "Error: No JSON files found in $PRED_CONFIGS_DIR"
    exit 1
fi

echo "Found ${#ckplist[@]} config files in $PRED_CONFIGS_DIR"

for ckp in ${ckplist[@]}
do
    echo "Processing: ${ckp}"

    if [[ "$ckp" == *"roberta"* ]]; then
        echo "Running with RoBERTa model"
        python -u ./run_modeling_roberta.py ${ckp}
    elif [[ "$ckp" == *"pmbert"* ]]; then
        echo "Running with PmBERT model"
        python -u ./run_modeling_bert.py ${ckp}
    else
        echo "Warning: Unknown model type in ${ckp}, skipping..."
    fi
done

