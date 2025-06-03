
## Overview
We propose CoopBert

## Results on STS Tasks

## Evaluation

To evaluate the models:

```sh
bash eval_only.sh [unsup-bert|sup-bert]
```
### Setup
'''sh
bash git clone -b temp https://github.com/brongs2/CoOp-BERT.git
'''
'''sh
bash cd CoOp-BERT
cd SentEval/data/downstream/
bash download_dataset.sh
cd -
cd ./data
bash download_wiki.sh
bash download_nli.sh
cd -
file_path = "/content/CoOp-BERT/SentEval/senteval/sts.py"

with open(file_path, "r") as f:
    lines = f.readlines()

with open(file_path, "w") as f:
    for line in lines:
        line = line.replace(
            'np.array([s.split() for s in sent1])[not_empty_idx]',
            '[s.split() for i, s in enumerate(sent1) if not_empty_idx[i]]'
        )
        line = line.replace(
            'np.array([s.split() for s in sent2])[not_empty_idx]',
            '[s.split() for i, s in enumerate(sent2) if not_empty_idx[i]]'
        )
        f.write(line)

print("sts modified")
'''


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
