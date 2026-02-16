export CUDA_VISIBLE_DEVICES="2"
python eval.py ./data_tagged/fold_0_train.json
python eval.py ./data_tagged/fold_0_val.json