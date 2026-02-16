export CUDA_VISIBLE_DEVICES="1"
python eval.py ./data_tagged/fold_0_train.json
python eval.py ./data_tagged/fold_0_val.json