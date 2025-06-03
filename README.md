
## Overview
We propose CoopBert

## Results on STS Tasks

## Evaluation

To evaluate the models:

```sh
bash eval_only.sh [unsup-bert|sup-bert]
```

### Train with CoOp

```sh
!CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_name_or_path bert-base-uncased \
  --train_file data/nli_for_simcse.csv \
  --output_dir result/coop-bert \
  --num_train_epochs 1 \
  --per_device_train_batch_size 128 \
  --learning_rate 1e-5 \
  --max_seq_length 32 \
  --temp 0.05 \
  --do_train \
  --preprocessing_num_workers 10 \
  --overwrite_output_dir \
  --use_coop \
  --coop_length 15
```

### Evaluate CoOp model

```sh
!CUDA_VISIBLE_DEVICES=0 python evaluation.py \
  --model_name_or_path result/coop-bert \
  --use_coop \
  --coop_length 15 \
  --pooler avg \
  --mode test
```
