'''
Main script for distributed training of PLDR-LLM v510Gi
v510Gi is intended for inference only for ablation studies. 
This module is provided for completeness since support for training PLDR-LLM v510 is available.
'''

import logging
logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s')
logger=logging.getLogger("Main")
logger.setLevel(logging.DEBUG)

import os
import random
import numpy as np
import argparse
import time
import torch

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

import pldr_run_model_v510Gi as pldr_run_model
import pldr_data_prep as pldr_data_prep

from pytz import timezone
pst_time=timezone('US/Pacific')
from datetime import datetime

def setup(rank, world_size, master_addr, master_port):
    os.environ['MASTER_ADDR'] = master_addr #eg. 'localhost'
    os.environ['MASTER_PORT'] = master_port #eg. '12345'
    # initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def fsdp_pldr_main(rank, world_size, args):
    master_addr, master_port=args.master_addr, args.master_port
    setup(rank, world_size, master_addr=master_addr, master_port=master_port)

    if rank==0:
        logger.info(f"{datetime.now(pst_time)}: LIST OF ARGUMENTS FOR THIS RUN:")
        for key, val in vars(args).items():
            logger.info(f"{key}: {val}")

    tok_model=args.tok_model
    context_length=args.context_length

    #PREPARE THE DATASET
    BATCH_SIZE=args.batch_size #16
    BUFFER_SIZE=args.buffer_size
    dataset_file=args.dataset_file
    dataset_column_label=args.dataset_column_label
    split_names=args.split_names
    load_dataset=args.load_dataset
    load_from_train=args.load_from_train
    split_style= args.split_style
    TRAIN_SAMPLE_INTERVAL=args.train_sample_interval # list: [start, end] eg. [0, 8000000]
    VAL_SAMPLE_SIZE=args.val_sample_size
    test_offset=args.test_offset
    MAX_LENGTH=args.context_length
    batch_agg_count=args.batch_agg_count
    padding_type=args.padding_type
    trust_remote_code=args.trust_remote_code
    shuffle_set=args.shuffle_set
    shuffle_seed=args.shuffle_seed

    if not shuffle_set:
        logger.info(f"(RANK {rank}):{datetime.now(pst_time)}: Dataset is not shuffled.")

    logger.info(f"(RANK {rank}):{datetime.now(pst_time)}: PREPARING DATA")
    start=time.time()
    inp_obj = pldr_data_prep.dist_pldr_data_prep(rank=rank,
                                            WORLD_SIZE=world_size,
                                            BUFFER_SIZE=BUFFER_SIZE,
                                            BATCH_SIZE = BATCH_SIZE,
                                            dataset_file=dataset_file,
                                            dataset_column_label=dataset_column_label,
                                            split_names=split_names,
                                            load_dataset=load_dataset,
                                            load_from_train=load_from_train,
                                            split_style= split_style,
                                            train_intvl= TRAIN_SAMPLE_INTERVAL,
                                            val_offset= VAL_SAMPLE_SIZE,
                                            test_offset=test_offset,
                                            tok_model=tok_model,
                                            shuffle_set=shuffle_set,
                                            shuffle_seed=shuffle_seed,
                                            MAX_LENGTH=MAX_LENGTH,
                                            batch_agg_count=batch_agg_count,
                                            padding_type=padding_type,
                                            trust_remote_code=trust_remote_code,
                                            )
    
    logger.info(f"(RANK {rank}):{datetime.now(pst_time)}: DATA PREPARED IN {(time.time()-start):.2f}s")

    epochs=args.epochs
    num_layers=args.num_layers
    num_heads=args.num_heads
    dk=args.dk
    warmup_steps=args.warmup_steps 
    train_batches_cnt=args.train_batches_cnt
    val_batches_cnt=args.val_batches_cnt
    lr_total_steps=train_batches_cnt
    learning_rate=args.learning_rate
    lr_alpha=args.lr_alpha
    adamw_decay=args.adamw_decay
    activation=F.silu
    disable_amp=args.disable_amp
    save_model_path=args.save_model_path
    auto_size_minimum=args.auto_size_minimum
    enable_batch_count=args.enable_batch_count
    
    if enable_batch_count:
        logger.info(f"(RANK {rank}):{datetime.now(pst_time)}: Starting batch count for train and validation batches.")
        for i, _ in enumerate(inp_obj.train_batches):
            cnt=i+1
            if cnt%50000==0:
                logger.info(f"(RANK {rank}):{datetime.now(pst_time)}: Iterated through {cnt} batches of batch size {BATCH_SIZE}")
        logger.info(f"(RANK {rank}):{datetime.now(pst_time)}: Total number of batches in train dataset in Rank {rank} is {cnt}")
        assert train_batches_cnt < cnt, f"Requested train batch coun {train_batches_cnt} is larger than available number of batches {cnt}"

        if inp_obj.val_batches is not None:
            for i, _ in enumerate(inp_obj.val_batches):
                cnt=i+1
                if cnt%1000==0:
                    logger.info(f"(RANK {rank}):{datetime.now(pst_time)}: Iterated through {cnt} batches of batch size {BATCH_SIZE}")
            logger.info(f"(RANK {rank}):{datetime.now(pst_time)}:Total number of batches in validation dataset in Rank {rank} is {cnt}")
            assert val_batches_cnt < cnt, f"Requested val batch count {val_batches_cnt} is larger than available number of batches {cnt}"


    #Derived parameters
    d_model= num_heads*dk
    dff=args.dff if args.dff is not None else int(np.floor(num_heads*dk*4*2/3))
    A_dff=args.Adff if args.Adff is not None else int(np.floor(4*dk*2/3))
    num_reslayerA=args.num_reslayerA
    num_denseA=args.num_denseA

    auto_size_minimum=args.auto_size_minimum
    disable_fsdp_mixed_precision=args.disable_fsdp_mixed_precision
    fsdp_cpu_offload=args.fsdp_cpu_offload
    fsdp_sharding_strategy=args.fsdp_sharding_strategy
    backward_prefetch=args.backward_prefetch
    save_type=args.save_type

    hyperparameter_dict= {
          "num_layers":num_layers,
          "d_model": d_model, 
          "num_heads": num_heads, 
          "dff": dff,
          "A_dff":A_dff,
          "num_reslayerA":num_reslayerA,
          "num_denseA":num_denseA,
          "input_vocab_size": inp_obj.tokenizer.vocab_size,
          "max_seq_len": context_length,
          "epochs":epochs, 
          "save_model_path": save_model_path,    
          "warmup_steps": warmup_steps,
          "lr_total_steps": lr_total_steps,
          "learning_rate": learning_rate,
          "lr_alpha": lr_alpha,
          "adamw_decay": adamw_decay,
          "activation": activation,
          "disable_amp": disable_amp,
          "auto_size_minimum": auto_size_minimum,
          "disable_fsdp_mixed_precision":disable_fsdp_mixed_precision,
          "fsdp_cpu_offload": fsdp_cpu_offload,
          "fsdp_sharding_strategy": fsdp_sharding_strategy,
          "backward_prefetch": backward_prefetch,
          "save_type":save_type
          }

    #INITIALIZE THE MODEL
    checkpoint_path=args.checkpoint_path
    load_ckpt=args.load_ckpt
    enable_full_dist_load=args.enable_full_dist_load
    is_train=args.is_train
    device=args.device
    logger.info(f"(RANK {rank}):{datetime.now(pst_time)}: INITIALIZING THE MODEL")
    e2e_obj=pldr_run_model.dist_pldr_model_e2e(rank=rank, world_size=world_size,
                                               inp_obj_src = inp_obj,
                                               checkpoint_path = checkpoint_path,
                                               hpdict=hyperparameter_dict,
                                               is_train=is_train,
                                               device=device,
                                               load_ckpt=load_ckpt, 
                                               enable_full_dist_load=enable_full_dist_load)
    logger.info(f"(RANK {rank}):{datetime.now(pst_time)}: MODEL INITIALIZED")
    
    #TRAIN THE MODEL
    verbose_freq = args.verbose_freq
    val_verbose_freq=args.val_verbose_freq
    chkpt_epochs=args.chkpt_epochs
    chkpt_batches= args.chkpt_batches
    
    
    logger.info(f"(RANK {rank}):{datetime.now(pst_time)}: TRAINING THE MODEL")
    train_loss, train_accuracy, val_loss, val_accuracy=e2e_obj.dist_train_model(train_batches=inp_obj.train_batches,
                                                                                train_batches_cnt=train_batches_cnt,
                                                                                val_batches_cnt=val_batches_cnt,
                                                                                val_batches=inp_obj.val_batches,
                                                                                chkpt_batches=chkpt_batches,
                                                                                chkpt_epochs=chkpt_epochs,
                                                                                verbose_freq=verbose_freq,
                                                                                val_verbose_freq=val_verbose_freq
                                                                                )
    

    logger.info(f"(RANK {rank}):{datetime.now(pst_time)}: MODEL TRAINING ENDED")
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Fully Sharded Distributed Parallel PLDR-LLM Training')
    
    #FSDP parameters
    parser.add_argument('--master_addr', type=str, default='localhost', metavar='ADDR',
                    help='Master Address for FSDP setup (Default: localhost)')
    parser.add_argument('--master_port', type=str, default=12345, metavar='PORT',
                    help='Master Port for FSDP setup (Default: 12345)')
    
    #DATASET PREP PARAMETERS
    parser.add_argument('--batch_size', type=int, default=16, metavar='BATCH_SIZE',
                    help='Batch Size per Rank (Default: 16)')
    parser.add_argument('--tok_model', type=str, default='"my_tokenizer_model.model"',
                          metavar='TOKENIZER', help='Path to SentencePiece Model File')   
    parser.add_argument('--context_length', type=int, default=1024, metavar='CONTEXT_LENGTH',
                    help='Context Length for PLDR-LLM (Default: 1024)')
    parser.add_argument('--train_sample_interval', type=int, nargs=2, default=[0, 8000000], metavar='SAMPLE SIZE',
                    help='Start and End indices/percent of train Samples from Dataset (Default: (as index) [0, 8000000])')
    parser.add_argument('--val_sample_size', type=int, default=64000, metavar='SAMPLE SIZE',
                    help='Number of Val Samples from Dataset (Default: 64000)')
    parser.add_argument('--buffer_size', type=int, default=20000, metavar='BUFFER',
                    help='DATA PREP:Buffer Size for Shuffling (Default: 20000)')
    parser.add_argument('--dataset_file', type=str, default='tiiuae/falcon-refinedweb', metavar='DATASET-FILE',
                    help='DATA PREP: Path to Dataset Repo (Default: tiiuae/falcon-refinedweb)')
    parser.add_argument('--dataset_column_label', type=str, default='content', metavar='COL-NAME',
                    help='DATA PREP: Dataset Sample Column Name (Default: content)')
    parser.add_argument('--split_names', type=str, nargs=3, default=None, metavar='SLIT-LIST',
                    help='DATA PREP: List of split names if different from [train, validation, test] (Default: None)')
    parser.add_argument('--load_dataset', action='store_true', default=False,
                    help='DATA PREP: Flag to allow to load and preprocess dataset (Default: False)')
    parser.add_argument('--load_from_train', action='store_true', default=False, 
                    help='DATA PREP: Flag to allow validation and test from tain dataset (Default: False)')
    parser.add_argument('--split_style', type=str, default='index', metavar='SPLIT-STYLE',
                    help='DATA PREP: Specify index or percent based partition (Default: index)')
    parser.add_argument('--test_offset', type=int, default=None, metavar='TEST-OFFSET',
                    help='DATA PREP: Offset value to slice test data (Default: None)')
    parser.add_argument('--shuffle_set', action='store_true', default=False,
                    help='DATA PREP: Flag to enable shuffling dataset (Default: False)')
    parser.add_argument('--shuffle_seed', type=int, default=None, metavar='SHUFFLE_SEED',
                    help='DATA PREP: randomization seed for shuffling dataset. (Default: None)')
    parser.add_argument('--batch_agg_count', type=int, default=100, metavar='BATCH-AGG-CNT',
                    help='DATA PREP: Multiplier to batchsize for concatenating samples (Default: 100)')
    parser.add_argument('--padding_type', type=str, choices=['pack', 'pad', 'nopad'], default='pack', metavar='PADDING-TYPE',
                    help='DATA PREP: Padding approach in concatenating and chunking samples:\
                    pack, pad, or nopad (Default: pack)')
    parser.add_argument('--trust_remote_code', action='store_true', default=False,
                    help='DATA PREP: Enable executing remote code (Default: False)')
    parser.add_argument('--enable_batch_count', action='store_true', default=False,
                    help='DATA PREP: Enable counting of train and validation batches (Default: False)')

    
    #MODEL PARAMETERS
    parser.add_argument('--num_layers', type=int, default=5, metavar='NUM-LAYERS',
                    help='MODEL PARAM: Number of Decoder Layers (Default: 5)')    
    parser.add_argument('--num_heads', type=int, default=14, metavar='NUM-HEADS',
                    help='MODEL PARAM: Number of Attention Heads in a Decoder Layer (Default: 14)')
    parser.add_argument('--dk', type=int, default=64, metavar='DK',
                    help='MODEL PARAM: Embedding layer dim per head: d_model=num_heads*dk (Default: 64)')
    parser.add_argument('--num_reslayerA', type=int, default=8, metavar='NUM-RESLAYERA',
                    help='MODEL PARAM: Number of residual layers (Default: 8)')
    parser.add_argument('--num_denseA', type=int, default=2, metavar='NUM-DENSEA',
                    help='MODEL PARAM: Number of gated linear units for one residual layer (Default: 2)')
    parser.add_argument('--dff', type=int, default=None, metavar='DFF',
                    help='MODEL PARAM: Gated Linear Unit size for feed forward network (Default: floor(4*2/3*d_model))')
    parser.add_argument('--Adff', type=int, default=None, metavar='ADFF',
                    help='MODEL PARAM: Gated Linear Unit size for reslayer (Default: floor(4*2/3*dk))')
    parser.add_argument('--epochs', type=int, default=1, metavar='EPOCHS',
                    help='MODEL PARAM: Number of training epochs to run (Default: 1)')
    parser.add_argument('--save_model_path', type=str, default='my_model_save_folder', metavar='SAVE-PATH',
                    help='MODEL PARAM: Folder name to save a specific model checkpoint files (Default: my_model_save_folder)')
    parser.add_argument('--warmup_steps', type=int, default=2000, metavar='WARMUP-STEPS',
                    help='MODEL PARAM: LR cosine scheduler linear warmup steps count (Default: 2000)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, metavar='LEARNING-RATE',
                    help='MODEL PARAM: Maximum Learning Rate (Default: 1e-3)')
    parser.add_argument('--lr_alpha', type=float, default=0.1, metavar='SAVE-PATH',
                    help='MODEL PARAM: Final learning rate as percentage of maximum learning rate (Default: 0.1)')
    parser.add_argument('--adamw_decay', type=float, default=0.1, metavar='SAVE-PATH',
                    help='MODEL PARAM: Decay parameter for AdamW Optimizer (Default: 0.1)')
    parser.add_argument('--auto_size_minimum', type=int, default=None, metavar='AUTO-SIZE-MINIMUM',
                    help='MODEL PARAM: Min numel size for FSDP auto size wrap policy, specify size to enable (Default: None)')
    parser.add_argument('--disable_amp', action='store_true', default=False,
                    help='MODEL PARAM: Enable and set automatic mixed precision for the model:bfloat16 (Default: True)')
    parser.add_argument('--disable_fsdp_mixed_precision', action='store_true', default=False,
                    help='MODEL PARAM: Enable fsdp native mixed precision for the model:bfloat16 or float16 (Default: True)')
    parser.add_argument('--fsdp_cpu_offload', action='store_true', default=False,
                    help='MODEL PARAM: Enable cpuoffload with offload params true (Default: False)')
    parser.add_argument('--fsdp_sharding_strategy', type=str, 
                        choices=["FULL_SHARD", "HYBRID_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "_HYBRID_SHARD_ZERO2"], default="FULL_SHARD",
                        metavar="SHARDING-STRATEGY", help='FSDP SHARDING STRATEGY (Default: "FULL_SHARD")')
    parser.add_argument('--backward_prefetch', type=str, 
                        choices=["PRE", "POST", "NOPREFETCH"], default="PRE",
                        metavar="BACKWARD-PREFETCH", help='Backward Prefetch option (Default: "PRE")')
    
    #TRAIN PARAMETERS
    parser.add_argument('--checkpoint_path', type=str, default='./my_model_checkpoints', metavar='SAVE-PATH',
                    help='TRAIN PARAM: File path location to save all model checkpoint files (Default: my_model_checkpoints)')
    parser.add_argument('--chkpt_batches', type=int, nargs='+', default=None, metavar='CHKPT-BATCHES',
                    help='TRAIN PARAM: List of batch numbers to get a model checkpoint saved at. (Default: None)')
    parser.add_argument('--chkpt_epochs', type=int, nargs='+', default=None, metavar='CHKPT-EPOCHS',
                    help='TRAIN PARAM: List of batch numbers to get a model checkpoint saved at. (Default: None)')
    parser.add_argument('--verbose_freq', type=int, default=2000, metavar='VERBOSE-FREQ',
                    help='TRAIN PARAM: Frequency at which to show metric updates (Default: 2000)')
    parser.add_argument('--val_verbose_freq', type=int, default=12000, metavar='VAL-VERBOSE-FREQ',
                    help='TRAIN PARAM: Frequency at which to evaluate validation data (Default: 12000)')
    parser.add_argument('--save_type', type=str, choices=["both", "dcp", "torch"], default="both", metavar="SAVE-TYPE",
                        help='TRAIN PARAM: Choose to save distributed (dcp) or non-distributed (torch) format(Default: "both")')
    parser.add_argument('--enable_full_dist_load', action='store_true', default=False,
                        help='TRAIN PARAM: Choose to load distributed (dcp) checkpoint as distributed or non-distributed (Default: False)')
    parser.add_argument('--load_ckpt', type=str, default=None, metavar="LOAD-CHKPT",
                        help='TRAIN PARAM: Provide a path to a torch save file or dcp save folder (Default: None)')
    parser.add_argument('--train_batches_cnt', type=int, default=None, metavar='TOTAL-STEPS',
                    help='MODEL PARAM: Number of batches to train per rank (also lr_total_steps) (Default: None)')
    parser.add_argument('--val_batches_cnt', type=int, default=None, metavar='VAL-VERBOSE-FREQ',
                    help='TRAIN PARAM: Validation batch count to evaluate per rank (Default: 2000)')
    parser.add_argument('--is_train', action='store_true', default=False,
                        help='TRAIN PARAM: Enable training of model (Default: False)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default="cuda", metavar="DEVICE",
                        help="TRAIN PARAM: Device to load the model and train/eval (Default: cuda)")
    
    args = parser.parse_args()

    WORLD_SIZE = torch.cuda.device_count()
    mp.spawn(fsdp_pldr_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)
