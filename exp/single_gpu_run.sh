export CUDA_VISIBLE_DEVICES="1,2,0,3"
CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=test_data/my-gpt2_text_document
export FP16_TRAINING=True
GPT_ARGS="--num-layers 2 \
          --hidden-size 1024 \
          --num-attention-heads 16 \
          --seq-length 1024 \
          --max-position-embeddings 1024 \
          --micro-batch-size 1 \
          --global-batch-size 2 \
          --lr 0.00015 \
          --lr-decay-iters 320000 \
          --lr-decay-style cosine \
          --vocab-file $VOCAB_FILE \
          --merge-file $MERGE_FILE \
          --lr-warmup-fraction .01 \
          --fp16 \
          --train-iters 2000"

OUTPUT_ARGS="--log-interval 1 \
           --save-interval 500 "
 
#       --save-interval 500 \
#       --eval-interval 100 \
#       --eval-iters 10 \
#       --checkpoint-activations"

python ../pretrain_gpt.py \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
