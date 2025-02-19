#!/bin/bash

SRC_PATH=$( realpath "../../../src" )
export PYTHONPATH=$SRC_PATH

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export HF_DATASETS_CACHE="path/to/huggingface/datasets/cache"

echo HF_DATASETS_CACHE=$HF_DATASETS_CACHE
echo PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF
echo PYTHONPATH=$PYTHONPATH

MASTER_ADDR='localhost'
MASTER_PORT=12345

# DATA PREP PARAMETERS
TOKEN_MODEL="path/to/sentencepiece/tokenizer/file"
CHECKPOINT_PATH="path/to/checkpoint/folder"

CONTEXT_LENGTH=1024
BATCH_SIZE=16
BUFFER_SIZE=20000
DATASET_FILE='tiiuae/falcon-refinedweb'
DATASET_COLUMN_LABEL='content'
declare -a TRAIN_SAMPLE_INTERVAL=(16000000 32000000)
VAL_SAMPLE_SIZE=160000
TRAIN_BATCHES_CNT=250000
VAL_BATCHES_CNT=2000
SPLIT_STYLE='index'
BATCH_AGG_COUNT=100
PADDING_TYPE='pack'

# MODEL PARAMETERS
NUM_LAYERS=5
NUM_HEADS=14
DK=64
NUM_RESLAYERA=8
NUM_DENSEA=2
EPOCHS=1
WARMUP_STEPS=2000
LEARNING_RATE=3e-4
LR_ALPHA=0.1
ADAMW_DECAY=0.1
FSDP_SHARDING_STRATEGY="HYBRID_SHARD"
BACKWARD_PREFETCH="PRE"
GCACHELSTFILE="/path/to/predefined/Gcache/pkl/file"

# TRAIN PARAMETERS
declare -a CHKPT_BATCHES=(4000 12000 31250 62500 93750 125000 175000 225000)
VERBOSE_FREQ=2000
VAL_VERBOSE_FREQ=12000

SAVE_MODEL_PATH=my-sample-model-pldrllm-v510G
SAVE_TYPE="torch"
DEVICE='cuda'
logfile=$( realpath "../../../train-logs/${SAVE_MODEL_PATH}.log" )
runfile=$( realpath "../dist_pldr_v510G_train_main.py" )

python $runfile  --master_addr=$MASTER_ADDR \
                 --master_port=$MASTER_PORT \
                 --batch_size=$BATCH_SIZE \
                 --tok_model=$TOKEN_MODEL \
                 --context_length=$CONTEXT_LENGTH \
                 --train_sample_interval "${TRAIN_SAMPLE_INTERVAL[@]}" \
                 --val_sample_size=$VAL_SAMPLE_SIZE \
                 --buffer_size=$BUFFER_SIZE \
                 --dataset_file=$DATASET_FILE \
                 --dataset_column_label=$DATASET_COLUMN_LABEL \
                 --load_dataset \
                 --load_from_train \
                 --split_style=$SPLIT_STYLE \
                 --batch_agg_count=$BATCH_AGG_COUNT \
                 --padding_type=$PADDING_TYPE \
                 --trust_remote_code \
                 --num_layers=$NUM_LAYERS \
                 --num_heads=$NUM_HEADS \
                 --dk=$DK \
                 --gcachelst=$GCACHELSTFILE \
                 --epochs=$EPOCHS \
                 --save_model_path=$SAVE_MODEL_PATH \
                 --warmup_steps=$WARMUP_STEPS \
                 --train_batches_cnt=$TRAIN_BATCHES_CNT \
                 --val_batches_cnt=$VAL_BATCHES_CNT \
                 --learning_rate=$LEARNING_RATE \
                 --lr_alpha=$LR_ALPHA \
                 --adamw_decay=$ADAMW_DECAY \
                 --checkpoint_path=$CHECKPOINT_PATH \
                 --chkpt_batches "${CHKPT_BATCHES[@]}" \
                 --fsdp_sharding_strategy=$FSDP_SHARDING_STRATEGY \
                 --backward_prefetch=$BACKWARD_PREFETCH \
                 --verbose_freq=$VERBOSE_FREQ \
                 --val_verbose_freq=$VAL_VERBOSE_FREQ \
                 --is_train \
                 --device=$DEVICE \
                 --save_type=$SAVE_TYPE 2>&1 | tee $logfile

                #other options that are not specified and use the default values
                # --enable_batch_count  
                # --fsdp_cpu_offload
                # --disable_amp
                # --disable_fsdp_mixed_precision
                # --split_names
                # --test_offset
                # --shuffle_set
                # --auto_size_minimum
                # --chkpt_epochs
                # --enable_full_dist_load
                # --dff

