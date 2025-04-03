#!/bin/bash

python train_lstm.py \
    --data_dir ./data/porto \
    --data_file_name porto_processed \
    --out_dir ./results/lstm/models \
    --log_file ./results/lstm/logs/train.log \
    --batch_size 64 \
    --embedding_dim 128 \
    --hidden_dim 256 \
    --n_layers 2 \
    --dropout 0.2 \
    --lr 3e-4 \
    --beta1 0.9 \
    --beta2 0.99 \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --max_iters 50 \
    --eval_interval 5 \
    --log_interval 200 \
    --warmup_iters 10 \
    --threshold_percentile 95