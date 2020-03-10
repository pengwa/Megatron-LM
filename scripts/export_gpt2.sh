#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

python pretrain_gpt2.py \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --batch-size 8 \
       --seq-length 1024 \
       --vocab-size 50257 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 
#       --warmup .01 \
#       --checkpoint-activations \
#       --fp16


python pretrain_gpt2.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 8 \
       --seq-length 1024 \
       --vocab-size 50257 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 


python pretrain_gpt2.py \
       --num-layers 36 \
       --hidden-size 1280 \
       --num-attention-heads 20 \
       --batch-size 8 \
       --seq-length 1024 \
       --vocab-size 50257 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 
#

python pretrain_gpt2.py \
       --num-layers 48 \
       --hidden-size 1600 \
       --num-attention-heads 25 \
       --batch-size 8 \
       --seq-length 1024 \
       --vocab-size 50257 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 

python pretrain_gpt2.py \
       --num-layers 40 \
       --hidden-size 1536 \
       --num-attention-heads 16 \
       --batch-size 8 \
       --seq-length 1024 \
       --vocab-size 50257 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 

python pretrain_gpt2.py \
       --num-layers 54 \
       --hidden-size 1920 \
       --num-attention-heads 20 \
       --batch-size 8 \
       --seq-length 1024 \
       --vocab-size 50257 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 

python pretrain_gpt2.py \
       --num-layers 64 \
       --hidden-size 2304 \
       --num-attention-heads 24 \
       --batch-size 8 \
       --seq-length 1024 \
       --vocab-size 50257 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 

python pretrain_gpt2.py \
       --num-layers 72 \
       --hidden-size 3072 \
       --num-attention-heads 32 \
       --batch-size 8 \
       --seq-length 1024 \
       --vocab-size 50257 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 


#
set +x
