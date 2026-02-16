#! /bin/bash

if true; then
type="_with_extra"
sfs=(0.02)
lrs=(3e-05)
bss=(16)
fidxs=(0)
for sf in ${sfs[@]}
do
for lr in ${lrs[@]}
do
for bs in ${bss[@]}
do
for fidx in ${fidxs[@]}
do
python -u ./run_modeling_bert.py ./configs${type}/config_pmbert_ls${sf}_split_${fidx}_${bs}_${lr}.json 
python -u ./run_modeling_roberta.py ./configs${type}/config_roberta_ls${sf}_split_${fidx}_${bs}_${lr}.json 
done
done
done
done
fi


