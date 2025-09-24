#!/bin/bash

SCRIPT="run_our.py"

run_arxiv() {
    echo "noise_level $1"
    echo "model $2"
    python ${SCRIPT} --log_steps 1 \
                    --eval_steps 1 \
                    --type $2 \
                    --num_layers 3 \
                    --num_heads $4 \
                    --batch_size 10000 \
                    --num_workers 7 \
                    --dataset "ogbn-arxiv" \
                    --hidden_channels 256 \
                    --dropout 0.25 \
                    --lr 0.001 \
                    --epochs 100 \
                    --runs 1 \
                    --use_saint \
                    --num_steps 30 \
                    --walk_length 3 \
                    --use_layer_norm \
                    --use_residual \
                    --noise_level $1 \
                    --patient 20 \
                    --max_loss $3 \
                    --min_epoch 30 | tee -a "arxiv_noise_${1}_${2}_results.txt"

}
run_arxiv_reg() {
    echo "noise_level $1"
    echo "model $2"
    python ${SCRIPT} --log_steps 1 \
                    --eval_steps 1 \
                    --type $2 \
                    --num_layers 3 \
                    --num_heads $4 \
                    --batch_size 10000 \
                    --num_workers 7 \
                    --dataset "ogbn-arxiv" \
                    --hidden_channels 256 \
                    --dropout 0.25 \
                    --lr 0.001 \
                    --epochs 200 \
                    --runs 1 \
                    --use_saint \
                    --num_steps 30 \
                    --walk_length 3 \
                    --use_layer_norm \
                    --use_causal_reg\
                    --use_residual \
                    --noise_level $1 \
                    --patient 20 \
                    --max_loss $3 \
                    --min_epoch 30 | tee -a "arxiv_noise_${1}_${2}_results.txt"

}
run_mag() {
    echo "noise_level $1"
    echo "model $2"
    python ${SCRIPT} --log_steps 1 \
                     --eval_steps 1 \
                     --type $2 \
                     --num_layers 2 \
                     --num_heads $4 \
                     --batch_size 10000 \
                     --num_workers 7 \
                     --dataset "ogbn-mag" \
                     --hidden_channels 256 \
                     --dropout 0.5 \
                     --lr 0.001 \
                     --epochs 100 \
                     --runs 10 \
                     --use_layer_norm \
                     --use_causal_reg\
                     --use_residual \
                     --noise_level $1 \
                     --patient $5 \
                     --max_loss $3 \
                     --min_epoch 0 | tee -a "mag_noise_${1}_${2}_results.txt"

}



run_arxiv_reg "0.5" "GAT" "100.0" "1"
run_arxiv "0.5" "GAT" "100.0" "1"

run_arxiv_reg "0.5" "GAT2" "3.0" "1"
run_arxiv "0.5" "GAT2" "1.3" "1"

