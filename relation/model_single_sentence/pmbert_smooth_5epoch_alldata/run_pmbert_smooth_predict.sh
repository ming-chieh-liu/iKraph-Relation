#! /bin/bash

if true; then
type="_with_extra"
ckplist=(`ls ./pred_configs${type}/*.json`)
for ckp in ${ckplist[@]}
do
    if [[ $ckp == *"roberta"* ]]; then
        python -u ./run_modeling_roberta.py ${ckp}
    else
        python -u ./run_modeling_bert.py ${ckp}
    fi
done
fi