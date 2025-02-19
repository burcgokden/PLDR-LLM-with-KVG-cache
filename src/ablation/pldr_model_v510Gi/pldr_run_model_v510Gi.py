'''
LLM from Power Law Decoder Representations v510Gi (PLDR-LLM v510Gi) with KV-cache and G-cache
v510G removes the deep residual layers and custom weight/biases that learn A, A_LM and G_LM
G_LM is provided as a predefined value during model initialization.
v510Gi is for inference only for ablation studies to transfer learnable weights from 
a PLDR-LLM v510 model and the G_LM from that model.
The weights and G_LM are extracted from PLDR-LLM-v510 and transferred to PLDR-LLM v510Gi
with this module. 
'''
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s')
logger=logging.getLogger("M")
logger.setLevel(logging.DEBUG)

import os
os.environ["KERAS_BACKEND"] = "torch"

import time
import math
import random
import numpy as np
import functools
import torch

# Uncomment below to set seed
# random.seed(1234)
# np.random.seed(1234)
# torch.manual_seed(1234)
# torch.cuda.manual_seed(1234)

from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import keras

import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (get_state_dict, 
                                                     get_model_state_dict,
                                                     set_state_dict,
                                                     set_model_state_dict,
                                                     StateDictOptions
                                                     )
from torch.distributed.fsdp import  (
                                     FullyShardedDataParallel as FSDP, 
                                     ShardingStrategy
                                     )
from torch.distributed.fsdp.fully_sharded_data_parallel import (
                                                                CPUOffload, 
                                                                BackwardPrefetch, 
                                                                MixedPrecision
                                                               )
from torch.distributed.fsdp.wrap import (
                                         size_based_auto_wrap_policy,
                                         transformer_auto_wrap_policy
                                        )

from pldr_model_v510 import (plgMultiHeadAttention, PLDR_DecoderLayer, PLDR_Model,
                                 PLDR_Decoder, GLUVariant, ResLayerA)
from power_law_attention_layer_v510 import plga_layer

import pldr_model_v510 as pldr_model
import pldr_model_v510Gi as pldr_model_infer
import common as cm


class dist_pldr_model_e2e:
    '''
    Trains and evaluates Large Language Model from Power Law Graph Decoder Representations
    '''

    def __init__(self, rank, world_size, inp_obj_src, hpdict=None,
                 checkpoint_path = "./pldr_checkpoints/", 
                 load_ckpt=None, device=None, is_train=True, enable_full_dist_load=None):
        '''
        Args:
        rank: Index of the device to run.
        world_size: Total number of ranks.
        inp_obj_src: Data preparation object to retrieve the tokenizer from.
        hpdict: Hyperparameter dictionary for the model.
        checkpoint_path: File path to create a train folder to save checkpoints and other data.
        load_ckpt: Path to a saved model checkpoitnt.
        device: Device to load the model on, cpu or cuda.
        is_train: Initialize for training.
        enable_full_dist_load: Load model in distributed setting onto multiple devices.
        '''
        self.rank=rank
        self.world_size=world_size
        self.is_train=is_train
        if device is None or device=='cuda' or device==f"cuda:{self.rank}":
            self.device=torch.device(f"cuda:{self.rank}")
            torch.cuda.set_device(self.device)
        elif device == 'cpu':
            self.device=torch.device('cpu')
        else:
            logger.warning("Unrecognized device type. Valid options are: cuda, cpu")


        if hpdict:
            self.hpdict = hpdict
        else:
            if self.rank==0:
                logger.info("USING DEFAULT HYPERPARAMETERS")
            self.hpdict={"num_layers": 5,
                         "d_model": int(14*64),
                         "num_heads": 14,
                         "dff": int(np.floor(14*64*4*2/3)),
                         "num_reslayerA":8,
                         "num_denseA":2,
                         "A_dff":170,
                         "input_vocab_size": inp_obj_src.tokenizer.get_vocab_size(),
                         "max_seq_len":4096,
                         "epochs":1,
                         "save_model_path": "default_pldr_model",
                         "warmup_steps": 2000, 
                         "lr_total_steps": 250000,
                         "learning_rate": 1e-3,
                         "lr_alpha":0.1,
                         "adamw_decay":0.1,
                         "activation":F.silu,
                         "device":'cuda',
                         "auto_size_minimum": None,
                         "disable_amp":False,
                         "disable_fsdp_mixed_precision":False,
                         "fsdp_cpu_offload":False,
                         "fsdp_sharding_strategy":"FULL_SHARD",
                         "backward_prefetch":"PRE",
                         "save_type": "torch"
            }
        if self.rank==0:
            logger.info("MODEL HYPERPARAMETERS:")
            for key,val in self.hpdict.items():
                logger.info(f"{key}: {val}")

        self.tokenize=inp_obj_src.tokenize
        self.detokenize=inp_obj_src.detokenize

        self.save_type=hpdict["save_type"]
        self.col_name='sample'

        self.pldr_model = pldr_model.PLDR_Model(
                            num_layers=self.hpdict["num_layers"],
                            d_model=self.hpdict["d_model"],
                            num_heads=self.hpdict["num_heads"],
                            dff=self.hpdict["dff"],
                            input_vocab_size=self.hpdict["input_vocab_size"],
                            A_dff=self.hpdict["A_dff"],
                            num_reslayerA=self.hpdict["num_reslayerA"],
                            num_denseA=self.hpdict["num_denseA"],
                            max_seq_len=self.hpdict["max_seq_len"],                            
                            activation=hpdict["activation"],
                            device=self.device
                        )
        
        if is_train or os.path.isdir(load_ckpt if load_ckpt is not None else ""):

            #train parameters
            self.loss_function=masked_loss_function
            self.use_amp=not self.hpdict["disable_amp"]
            self.use_fsdp_mp=not self.hpdict["disable_fsdp_mixed_precision"]
            self.min_num_params=hpdict["auto_size_minimum"]
            self.fsdp_cpu_offload=hpdict["fsdp_cpu_offload"]
            self.fsdp_sharding_strategy=hpdict["fsdp_sharding_strategy"]
            self.backward_prefetch=hpdict["backward_prefetch"]

            backwardprefetches={"PRE": BackwardPrefetch.BACKWARD_PRE,
                                "POST": BackwardPrefetch.BACKWARD_POST,
                                "NOPREFETCH":None
                                }

            sharding_strategies={"FULL_SHARD": ShardingStrategy.FULL_SHARD,
                                "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
                                "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
                                "_HYBRID_SHARD_ZERO2":ShardingStrategy._HYBRID_SHARD_ZERO2,
                                "NO_SHARD": ShardingStrategy.NO_SHARD
                                }
            
            if self.fsdp_cpu_offload:
                if self.rank==0:
                    logger.info("using cpu offload")
                cpu_offload=CPUOffload(offload_params=True)
            else:
                cpu_offload=None
                
            if self.rank==0:
                logger.info(f"USING MIXED PRECISION ARITHMETIC: {self.use_amp}")
                logger.info(f"USING FSDP NATIVE MIXED PRECISION ARITHMETIC: {self.use_fsdp_mp}")
                logger.info(f"USING FSDP SHARDING STRATEGY: {self.fsdp_sharding_strategy}")


            if self.min_num_params:
                if self.rank==0:
                    logger.info("Using size based auto-wrap policy")
                auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, 
                                                        min_num_params=self.min_num_params)
            else:
                if self.rank==0:
                    logger.info("Using transformer module based auto-wrap policy")
                auto_wrap_policy = functools.partial(
                                                            transformer_auto_wrap_policy,
                                                            transformer_layer_cls={
                                                                # PLDR_Model,
                                                                # PLDR_Decoder,
                                                                PLDR_DecoderLayer,
                                                                # plgMultiHeadAttention,
                                                                #  plga_layer,
                                                                #  ResLayerA,
                                                                # GLUVariant,
                                                                }
                                                                )

            if self.use_fsdp_mp:
                mp_policy = MixedPrecision(
                                            param_dtype=torch.bfloat16,
                                            reduce_dtype=torch.bfloat16,
                                            buffer_dtype=torch.bfloat16,
                                            cast_forward_inputs=True,
                                        )
            else:
                mp_policy=None

            self.dist_pldr_model = FSDP(self.pldr_model, auto_wrap_policy=auto_wrap_policy,
                                        cpu_offload=cpu_offload,
                                        mixed_precision=mp_policy,
                                        sharding_strategy= sharding_strategies[self.fsdp_sharding_strategy],
                                        backward_prefetch = backwardprefetches[self.backward_prefetch], 
                                        device_id=self.device
                                        )
        
            if self.rank==0:
                logger.info("Using AdamW optimizer with Linear Warm Up Cosine Annealing Schedule.")
            self.learning_rate=self.hpdict["learning_rate"]

            self.optimizer=torch.optim.AdamW(self.dist_pldr_model.parameters(), lr=self.learning_rate, betas=(0.9, 0.95),
                                            eps=1e-5, weight_decay=self.hpdict["adamw_decay"])
            
            self.scheduler=LinearWarmupCosineLRSchedule(self.optimizer, total_steps=self.hpdict["lr_total_steps"],
                                                            warmup_steps=self.hpdict["warmup_steps"],
                                                            alpha=self.hpdict["lr_alpha"],
                                                            last_epoch=-1)
            
            #For mixed precision with torch.autocast and gradient scaling.
            self.scaler= torch.amp.GradScaler(enabled=self.use_amp)

        if is_train:
            self.checkpoint_path = checkpoint_path
            self.train_ckpt_path=os.path.join(self.checkpoint_path, "train", self.hpdict["save_model_path"])
            if self.rank==0:
                if not os.path.isdir(self.train_ckpt_path):
                    logger.info(f"Creating train ckpt dir: {self.train_ckpt_path}")
                    os.makedirs(self.train_ckpt_path)
                cm.pklsave(os.path.join(self.train_ckpt_path, self.hpdict["save_model_path"] + "_hparams.pkl"), self.hpdict)

        if load_ckpt is not None:
            if os.path.isdir(load_ckpt): #dcp
                if rank==0:
                    logger.info("Attempting to restore the checkpoint path specified...")
                if enable_full_dist_load:
                    self.load_dcp_model(load_ckpt)
                else:
                    self.load_dcp_model_nondist(load_ckpt)
                if rank==0:
                    logger.info(f"Checkpoint restored at {load_ckpt}")
            elif os.path.isfile(load_ckpt): #torch
                if self.rank==0:
                    logger.info("Attempting to restore the checkpoint path specified...")
                self.load_model(load_ckpt, self.pldr_model)
                if self.rank==0:
                    logger.info(f"Checkpoint restored at {load_ckpt}")
            else:
                if self.rank==0:
                    logger.warning(f"Not a valid format to load a model: {load_ckpt}")

            #Generate G tensor values for each layer
            self.Gcachelst=self.generate_Glst(sentence="", temperature=1.0, 
                                              top_k=1, top_p=1.0, max_length=1, 
                                              save_att=None, save_Gcache=None)
            
            #inference model is loaded if G tensor can be generated.
            self.pldr_model_infer = pldr_model_infer.PLDR_Model(
                                num_layers=self.hpdict["num_layers"],
                                d_model=self.hpdict["d_model"],
                                num_heads=self.hpdict["num_heads"],
                                dff=self.hpdict["dff"],
                                Gcachelst=self.Gcachelst,
                                input_vocab_size=self.hpdict["input_vocab_size"],
                                max_seq_len=self.hpdict["max_seq_len"],                            
                                activation=hpdict["activation"],
                                device=self.device
                        )
            #load remaining network weights to pldr_model_inder
            self.xferlayers2infer()
        else:
            if self.rank==0:
                logger.info("Checkpoint restoration skipped.")
        
    
    def xferlayers2infer(self):
        xfer_dict=dict()
        for layer_name in self.pldr_model.state_dict().keys():
            layers2exclude=["reslayerAs", "plgatt_layer", "mha1.layernorm1"]
            if not any(s in layer_name for s in layers2exclude):
                xfer_dict[layer_name]=self.pldr_model.state_dict()[layer_name]
        self.pldr_model_infer.load_state_dict(xfer_dict, strict=True)
    
    def generate_Glst(self, sentence="", temperature=1.0, 
                      top_k=1, top_p=1.0, enable_kvcache=True, 
                      enable_Gcache=True, Gcachelst_init=None,
                      max_length=1, save_att=None, save_Gcache=None):
        '''
        This method returns a list of layer_num G deductive output tensors with
        size [1, num_head, dk, dk]
        '''
        _, att_weights, _=self.generate_text(sentence, temperature=temperature, 
                                                    top_k=top_k, top_p=top_p,
                                                    enable_kvcache=enable_kvcache, enable_Gcache=enable_Gcache, 
                                                    Gcachelst_init=Gcachelst_init, max_length=max_length, 
                                                    save_att=save_att)
        Glst=[t[4] for t in att_weights]
        if save_Gcache is not None:
            cm.pklsave(save_Gcache, Glst)
        return Glst
   
    def save_dcp_model(self, chkpt_dir):
        '''
        This method loads from a distributed checkpoint dir onto a distributed
        model on multiple ranks. AppState class defines helper classes to load state
        for distributed checkpointig. 
        '''        
        model_state_dict={ "app": AppState(self.dist_pldr_model, optimizer=None) }
        dcp.save(model_state_dict, checkpoint_id=chkpt_dir)
        return chkpt_dir
    
    def load_dcp_model(self, chkpt_dir):
        '''
        This method loads from a distributed checkpoint dir onto a distributed
        model on multiple ranks. AppState class defines helper classes to load state
        for distributed checkpointig. 
        '''
        model_state_dict={ "app": AppState(self.pldr_model, optimizer=None)}
        dcp.load(state_dict=model_state_dict, checkpoint_id=chkpt_dir)
        self.dist_pldr_model.load_state_dict(model_state_dict)
    
    def load_dcp_model_nondist(self, chkpt_dir):
        '''
        This method loads from distributed checkpoint dir into a non-distributed model.
        Note the model state dict is initialized from non-distributed pldr model.
        '''
        model_state_dict=self.pldr_model.state_dict()
        dcp.load(state_dict=model_state_dict, checkpoint_id=chkpt_dir)
        self.dist_pldr_model.load_state_dict(model_state_dict)


    def fsdp_save_model(self, save_path, loss=None, accuracy=None):
        '''
        This method moves model statedict to cpu through gathering in rank 0 to avoid 
        possible OOM for models that may  not fit in a single GPU memory.
        This model saves in the format same as single gpu model is saved.
        '''

        cpu_model_state=get_model_state_dict(self.dist_pldr_model,
                                             options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        
        if self.rank==0:
            loss=self.train_loss.result().item()
            accuracy=self.train_accuracy.result().item()
            dict_to_save={
            'model_state_dict': cpu_model_state,
            'loss': loss,
            'accuracy':accuracy
            } 
            torch.save(dict_to_save, save_path)
        return save_path

    @staticmethod
    def save_model(model_state_dict, save_path, optimizer_state_dict=None, loss=None, accuracy=None):
        '''
        Saves a model trained on a single gpu without any parallelization.
        '''
        dict_to_save={
                      'model_state_dict': model_state_dict,
                      'optimizer_state_dict': optimizer_state_dict,
                      'loss': loss,
                      'accuracy':accuracy
                      }
        torch.save(dict_to_save, save_path)
        return save_path
    
    @staticmethod
    def load_model(load_path, model, optimizer=None):
        '''
        Loads a model trained on a single gput without any parallelization.
        '''
        chkpt=torch.load(load_path, weights_only=True)
        model.load_state_dict(chkpt['model_state_dict'])
        if optimizer and chkpt['optimizer_state_dict']:
            optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        else:
            logger.info("Optimizer not available to load")

        return model, optimizer, chkpt['loss'], chkpt['accuracy']
    
    def save_chkpts(self, batch_cnt, epoch, save_cnt):
        '''
        This method chooses the type of saving based on save_type chosen during training setup.
        '''

        save_path, dcp_save_dir=None, None
        if self.save_type=="both":
            save_path=os.path.join(self.train_ckpt_path, f"{self.hpdict['save_model_path']}_ep{epoch+1}_bn{batch_cnt}_bs{save_cnt}.pth")
            dcp_save_dir=os.path.join(self.train_ckpt_path, f"{self.hpdict['save_model_path']}_ep{epoch+1}_bn{batch_cnt}_bs{save_cnt}_dcp")
            ckpt_save_path = self.fsdp_save_model( save_path=save_path)
            dcp_ckpt_save_dir = self.save_dcp_model(dcp_save_dir)
            if self.rank==0:               
                logger.info(f'Saving train checkpoint for batch {batch_cnt} in epoch {epoch+1} at {ckpt_save_path}')
                logger.info(f'Saving DCP train checkpoint for batch {batch_cnt} in epoch {epoch+1} at {dcp_ckpt_save_dir}')
        elif self.save_type=="dcp":
            dcp_save_dir=os.path.join(self.train_ckpt_path, f"{self.hpdict['save_model_path']}_ep{epoch+1}_bn{batch_cnt}_bs{save_cnt}_dcp")
            dcp_ckpt_save_dir = self.save_dcp_model(dcp_save_dir)
            if self.rank==0:
                logger.info(f'Saving DCP train checkpoint for batch {batch_cnt} in epoch {epoch+1} at {dcp_ckpt_save_dir}')
        elif self.save_type=="torch":
            save_path=os.path.join(self.train_ckpt_path, f"{self.hpdict['save_model_path']}_ep{epoch+1}_bn{batch_cnt}_bs{save_cnt}.pth")
            ckpt_save_path = self.fsdp_save_model(save_path=save_path)
            if self.rank==0:
                logger.info(f'Saving train checkpoint for batch {batch_cnt} in epoch {epoch+1} at {ckpt_save_path}')
        else:
            logger.warning("Unknown save type, saving checkpoint skipped.")
            
        return save_path, dcp_save_dir

    @staticmethod
    def view_state_dict(model):
        '''shows the model state_dict() contents'''
        for var_name in model.state_dict():
            print(var_name)

    @staticmethod
    def view_modules(model):
        '''shows modules in model'''
        for module in model.modules():
            print(module)

    @staticmethod
    def count_model_params(model):
        '''counts the model parameters'''
        param_cnt=sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f"Total number of parameters are: {param_cnt}")

        return param_cnt
    
    def create_masks(self, inp):
        '''
        inp: tensor of shape [batch_size, seq_len]
        Create masks for decoder layer for pldr model.
        Used in the attention block in the decoder.
        It is used to pad and mask future tokens in the input received by the decoder.
        '''
        look_ahead_mask = self.create_look_ahead_mask(inp.size()[1])
        dec_target_padding_mask = self.create_padding_mask(inp)
        combined_mask = torch.maximum(dec_target_padding_mask, look_ahead_mask)

        return combined_mask


    def create_padding_mask(self, seq):
        '''
        inp: tensor of shape [batch_size, seq_len]
        Create a mask for padding in the input for decoder.
        '''
        seq = torch.eq(seq,0)
        seq=seq.to(device=self.device, dtype=torch.float32)

        return seq[:, None, None, :]  # (batch_size, 1, 1, seq_len)
    

    def create_look_ahead_mask(self, size):
        '''
        The values that remain as 1 are multiplied with a small number
        so these entries vanish in attention calculation.
        '''
        mask = 1 - torch.tril(torch.ones((size, size)))
        return mask.to(device=self.device)


    def dist_train_step(self, inp):
        '''
        Train step for single token for PLDR-LLM with pretrain data input.
        inp: tensor of shape [batch_size, seq_len]
        '''

        #input is shifted by one to compare with predictions
        tar_inp = inp[:, :-1]
        tar_real = inp[:, 1:]

        combined_mask= self.create_masks(tar_inp)

        self.optimizer.zero_grad()

        #make model trainable
        self.dist_pldr_model.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            #Ensure caches are disabled by always having None value.
            predictions, _, _, _ = self.dist_pldr_model([tar_inp, combined_mask], kvcachelst=None, Gcachelst=None)
            loss=self.loss_function(tar_real, predictions)
            
        self.scaler.scale(loss).backward()

        # Unscale gradients first to clip them.
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_value_(self.dist_pldr_model.parameters(), clip_value=1.0, foreach=None)

        # self.optimizer.step()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        loss_ = loss.item()
        acc_ = self.accuracy_function(tar_real, predictions).item()

        with torch.no_grad():
            self.train_loss.update_state(loss_)
            self.train_accuracy.update_state(acc_)
            self.train_loss_epoch.update_state(loss_)
            self.train_accuracy_epoch.update_state(acc_)
            self.train_loss_one_epoch.update_state(loss_)
            self.train_accuracy_one_epoch.update_state(acc_)
        return loss_, acc_


    def dist_train_model(self, train_batches, train_batches_cnt=None, 
                         val_batches=None, val_batches_cnt=None,
                         chkpt_batches=None, chkpt_epochs=None,
                         verbose_freq=2000, val_verbose_freq=None):
        '''
        Method for training model for single and multiple epochs with batch indexed checkpointing.
        train_batches: batched dataset for pretraining.
        train_batches_cnt: Number of batches to train per epoch.
        val_batches: batched dataset for validation.
        val_batches_cnt: Number of batches to validate.
        chkpt_batches: A list of training steps at which checkpoints will be saved within a single epoch.
        chkpt_epochs: A list of training epochs at the end of which checkpoints will be saved.
        verbose_freq: how often train loss and metrics is saved and printed within e single epoch.
        val_verbose_freq: how often validation loss and metrics is saved and printed within e single epoch.
        '''

        self.train_loss = keras.metrics.Mean(name='train_loss')
        self.train_accuracy = keras.metrics.Mean(name='train_accuracy')
        self.val_loss = keras.metrics.Mean(name='val_loss')
        self.val_accuracy = keras.metrics.Mean(name='val_accuracy')
        self.train_loss_epoch = keras.metrics.Mean(name='train_loss_epoch')
        self.train_accuracy_epoch = keras.metrics.Mean(name='train_accuracy_epoch')
        self.train_loss_one_epoch = keras.metrics.Mean(name='train_loss_one_epoch')
        self.train_accuracy_one_epoch = keras.metrics.Mean(name='train_accuracy_one_epoch')

        EPOCHS=self.hpdict["epochs"]
        if self.rank==0:
            logger.info(f"Train checkpoints are at epochs: {chkpt_epochs}")
            logger.info(f"Batch Train checkpoints are at batches: {chkpt_batches}")

        #initialize lists to collect loss and accuracy data per epoch
        train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst=[],[],[],[]
        trn_loss_epoch_lst, trn_acc_epoch_lst=[], []
        trn_loss_one_epoch_lst, trn_acc_one_epoch_lst=[], []

        steps_count=0
        batch_save_cnt=0
        self.train_loss_epoch.reset_state()
        self.train_accuracy_epoch.reset_state()
        ddp_trn_res= torch.zeros(6).to(self.device)
        ddp_val_res= torch.zeros(2).to(self.device)

        for epoch in range(EPOCHS):
            start = time.time()

            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            self.train_loss_one_epoch.reset_state()
            self.train_accuracy_one_epoch.reset_state()

            iter_ds=iter(train_batches)
            for batch in range(train_batches_cnt):
                inp=next(iter_ds)[self.col_name].to(self.device)
                batch_cnt=batch+1
                steps_count+=1
                self.dist_train_step(inp)
                
                with torch.no_grad():
                    if batch_cnt % verbose_freq == 0 or batch==0:                   
                        time_so_far=time.time()-start
                        loss_batch=self.train_loss.result().item() 
                        acc_batch=self.train_accuracy.result().item() 
                        loss_epoch=self.train_loss_epoch.result().item() 
                        acc_epoch=self.train_accuracy_epoch.result().item() 
                        loss_one_epoch=self.train_loss_one_epoch.result().item() 
                        acc_one_epoch=self.train_accuracy_one_epoch.result().item()
                        ddp_trn_res=torch.tensor([loss_batch, acc_batch, loss_epoch, acc_epoch, loss_one_epoch, acc_one_epoch]).to(self.device)
                        dist.all_reduce(ddp_trn_res, op=dist.ReduceOp.SUM)
                        loss_batch, acc_batch, loss_epoch, acc_epoch, loss_one_epoch, acc_one_epoch=(ddp_trn_res/self.world_size).tolist()
                        
                        if self.rank==0:
                            logger.info(f"{time_so_far:.2f}s Epoch {epoch + 1} Batch {batch_cnt} Loss(G) {loss_epoch:.4f} Acc(G) {acc_epoch:.4f} "+
                                f"Loss(R) {loss_batch:.4f} Acc(R) {acc_batch:.4f} Loss(E) {loss_one_epoch:.4f} Acc(E) {acc_one_epoch:.4f} "+
                                f"LR {self.scheduler.get_last_lr()[0]:.4e}")
                        
                        train_loss_lst.append(loss_batch)
                        train_acc_lst.append(acc_batch)
                        trn_loss_epoch_lst.append(loss_epoch)
                        trn_acc_epoch_lst.append(acc_epoch)
                        self.train_loss.reset_state()
                        self.train_accuracy.reset_state()
                
                            
                    if val_verbose_freq is not None:
                        if batch_cnt % val_verbose_freq == 0:
                            self.validate_model(val_batches, val_batches_cnt)  
                            time_so_far=time.time()-start
                            val_lossv=self.val_loss.result().item()
                            val_accv=self.val_accuracy.result().item()
                            ddp_val_res=torch.tensor([val_lossv, val_accv]).to(self.device)
                            dist.all_reduce(ddp_val_res, op=dist.ReduceOp.SUM)
                            val_lossv, val_accv=(ddp_val_res/self.world_size).tolist()
                            val_loss_lst.append(val_lossv)
                            val_acc_lst.append(val_accv)
                            if self.rank==0:
                                logger.info(f"{time_so_far:.2f}s Epoch {epoch + 1} Batch {batch_cnt} Val Loss {val_lossv:.4f} Val Accuracy {val_accv:.4f}")
                        
                if chkpt_batches is not None:
                    if batch_cnt in chkpt_batches:
                        batch_save_cnt+=1
                        self.save_chkpts(batch_cnt, epoch, batch_save_cnt)

            if chkpt_epochs is not None:
                if (epoch + 1) in chkpt_epochs:
                    batch_save_cnt+=1
                    self.save_chkpts(batch_cnt, epoch, batch_save_cnt)

            with torch.no_grad():
                time_so_far=time.time()-start
                loss_batch=self.train_loss.result().item() 
                acc_batch=self.train_accuracy.result().item() 
                loss_epoch=self.train_loss_epoch.result().item() 
                acc_epoch=self.train_accuracy_epoch.result().item() 
                loss_one_epoch=self.train_loss_one_epoch.result().item()
                acc_one_epoch=self.train_accuracy_one_epoch.result().item()
                ddp_trn_res=torch.tensor([loss_batch, acc_batch, loss_epoch, acc_epoch, loss_one_epoch, acc_one_epoch]).to(self.device)
                dist.all_reduce(ddp_trn_res, op=dist.ReduceOp.SUM)
                loss_batch, acc_batch, loss_epoch, acc_epoch, loss_one_epoch, acc_one_epoch=(ddp_trn_res/self.world_size).tolist()
                if self.rank==0:
                    logger.info(f"Epoch {epoch + 1} Loss(G) {loss_epoch:.4f} Accuracy(G) {acc_epoch:.4f} "+
                        f"Loss(E) {loss_one_epoch:.4f} Accuracy(E) {acc_one_epoch:.4f}")
                if batch_cnt % verbose_freq != 0:
                    if self.rank==0:
                        logger.info(f"End of epoch batch count is {batch_cnt}. Appending end of epoch loss/accuracy")
                    trn_loss_epoch_lst.append(loss_epoch)
                    trn_acc_epoch_lst.append(acc_epoch)
                    trn_loss_one_epoch_lst.append(loss_one_epoch)
                    trn_acc_one_epoch_lst.append(acc_one_epoch)

                if val_batches is not None:
                    self.validate_model(val_batches, val_batches_cnt)
                    time_so_far=time.time()-start
                    val_lossv=self.val_loss.result().item()
                    val_accv=self.val_accuracy.result().item()
                    ddp_val_res=torch.tensor([val_lossv, val_accv]).to(self.device)
                    dist.all_reduce(ddp_val_res, op=dist.ReduceOp.SUM)
                    val_lossv, val_accv=(ddp_val_res/self.world_size).tolist()
                    val_loss_lst.append(val_lossv)
                    val_acc_lst.append(val_accv)
                    if self.rank==0:
                        logger.info(f"{time_so_far:.2f}s Epoch {epoch + 1} Batch {batch_cnt} Val Loss {val_lossv:.4f} Val Accuracy {val_accv:.4f}")
                if self.rank==0:
                    logger.info(f"Total number of steps elapsed: {steps_count}")
                    logger.info(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

        if self.rank==0:
            #save loss and accuracy data for train and validation runs
            cm.pklsave(self.train_ckpt_path+'/train_loss.pkl', train_loss_lst)
            cm.pklsave(self.train_ckpt_path+'/val_loss.pkl', val_loss_lst)
            cm.pklsave(self.train_ckpt_path+'/train_accuracies.pkl', train_acc_lst)
            cm.pklsave(self.train_ckpt_path+'/val_accuracies.pkl', val_acc_lst)
            cm.pklsave(self.train_ckpt_path+'/train_loss_epoch.pkl', trn_loss_epoch_lst)
            cm.pklsave(self.train_ckpt_path+'/train_loss_one_epoch.pkl', trn_loss_one_epoch_lst)
            cm.pklsave(self.train_ckpt_path+'/train_accuracies_epoch.pkl', trn_acc_epoch_lst)
            cm.pklsave(self.train_ckpt_path+'/train_accuracies_one_epoch.pkl', trn_acc_one_epoch_lst)

        batch_save_cnt+=1
        final_ckpt_save_path, final_ckpt_dcp_save_dir=self.save_chkpts(batch_cnt, epoch, batch_save_cnt)
        if self.rank==0:
            if final_ckpt_save_path:
                logger.info(f'Saving final train checkpoint for epoch {EPOCHS} at {final_ckpt_save_path}')
            if final_ckpt_dcp_save_dir:
                logger.info(f'Saving final train checkpoint for epoch {EPOCHS} at {final_ckpt_dcp_save_dir}')

        return train_loss_lst, train_acc_lst, val_loss_lst, val_acc_lst

    @torch.no_grad()
    def validate_step(self, inp):

        tar_inp = inp[:, :-1]
        tar_real = inp[:, 1:]

        combined_mask = self.create_masks(tar_inp)

        self.dist_pldr_model.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            predictions, _,_,_ = self.dist_pldr_model([tar_inp, combined_mask], kvcachelst=None, Gcachelst=None)
            loss = self.loss_function(tar_real, predictions)
        loss_=loss
        acc_=self.accuracy_function(tar_real, predictions)

        self.val_loss.update_state(loss_)
        self.val_accuracy.update_state(acc_)
        
        return loss_, acc_

    @torch.no_grad()
    def validate_model(self, val_batches, val_batches_cnt):
        '''
        This method runs the model on val dataset and returns loss and accuracy during training.
        Args:
            val_batches: the validation batches same size as train batches.
            val_batches_cnt: Number of batches to validate.
            epoch: current epoch.
        Returns:
            loss: the loss averaged over all batches.
            accuracy: the accuracy averaged over all batches.
        '''

        self.val_loss.reset_state()
        self.val_accuracy.reset_state()

        iter_valds=iter(val_batches)
        for _ in range(val_batches_cnt):
            inp=next(iter_valds)[self.col_name].to(self.device)
            self.validate_step(inp)

        return self.val_loss.result().item(), self.val_accuracy.result().item()
    
    @staticmethod
    def top_k_logits(logits, k):
        '''Top-k sampling'''
        if k == 0:
            # no truncation
            return logits

        def _top_k():
            values, _ = torch.topk(logits, k=k, sorted=True)
            min_values = values[:, -1, None]
            return torch.where(
                            logits < min_values,
                            torch.ones_like(logits, dtype=logits.dtype) * -1e10,
                            logits,
                           )
        return logits if k==0 else _top_k()

    def top_p_logits(self, logits, p):
        """Nucleus sampling"""
        batch, _ = logits.size()
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        indices = torch.stack([
                            torch.arange(0, batch).to(self.device),
                            # number of indices to include
                            torch.maximum(torch.sum((cumulative_probs <= p).to(torch.int32), dim=-1) - 1, torch.tensor(0)),
                            ], dim=-1).to(torch.int32).tolist()
        min_values = torch.tensor([sorted_logits[i[0],i[1]] for i in indices]).to(self.device)
        return torch.where(
            logits < min_values,
            torch.ones_like(logits) * -1e10,
            logits,
        )

    @torch.no_grad()
    def generate_text(self, sentence, 
                      temperature=1.0, top_k=0, top_p=1.0, 
                      enable_kvcache=True, enable_Gcache=True, Gcachelst_init=None,
                      max_length=50, save_att=None):
        '''
        Args:
            sentence: source sentence as input string.
            temperature: parameter to determine how deterministic the output is between (0,1]. 
                         Less deterministic on logits if temperature==1.
            top_k: value to select from top k largest logits, select all if k==0.
            top_p: cumulative probability threshold to select from logits for nucleus sampling. Select all if p == 1.
            enable_kvcache: enable caching of Key and Value tensors.
            enable_Gcache: enable caching of G_LM.
            Gcachelst_init: To provide a custom G_LM mainly for experiments when enable_Gcache is False.
            max_length: maximum number of iterations to run.
            save_att: path location to save attention weights.
        Returns:
            Predicted text, attention weights and max_length.
        '''
        assert 0.0 < temperature <=1.0, "set temperature between (0, 1]."
        assert 0.0 < top_p <=1.0, "set nucleus sampling probability between (0, 1], p=1 to skip top_p sampling."
        assert top_k >= 0, "set top_k above 0 or 0 to skip top_k sampling."

        sentence = torch.tensor(self.tokenize(sentence)).to(self.device)

        pldr_input = sentence
        end = torch.tensor(self.tokenize('')[0])
        output = pldr_input[:-1] if pldr_input.size()[0] > 1 else pldr_input # [seq_len]
        output = output[None, :] # [1, seq_len]
        seq_in=output
        att_weights=None
        kvcachelst=None
        Gcachelst=Gcachelst_init if Gcachelst_init is not None else None
        if Gcachelst is not None:
            Gcachelst=[[AW.to(self.device), avAp.to(self.device)] for AW, avAp in Gcachelst]
        cached=False

        for i in range(max_length):
            if not cached:
                combined_mask = self.create_masks(seq_in)
            else:
                combined_mask = None

            self.pldr_model.eval()
            predictions, _, att_weights, kvcachelst_nxt  = self.pldr_model([seq_in, combined_mask],
                                                           kvcachelst=kvcachelst,
                                                           Gcachelst=Gcachelst)

            if enable_kvcache:
                kvcachelst=kvcachelst_nxt
                cached=True     

            if enable_Gcache:
                    Gcachelst=[[t[0],t[4]] for t in att_weights]

            predictions = predictions[:, -1, :]  # (1, seq_len, vocab_size) -> (1, vocab_size)
            
            #temperature, top_k and nucleus sampling are stackable if needed.
            #scale logits for temperature sampling
            if temperature < 1:
                predictions = predictions/temperature

            #top_p sampling
            if top_p < 1:
                predictions=self.top_p_logits(logits=predictions, p=top_p)

            #top_k sampling
            if top_k > 0:
                predictions=self.top_k_logits(logits=predictions, k=top_k)
            

            predictions=torch.distributions.categorical.Categorical(logits=predictions) #(1, vocab_size)
            predicted_id=predictions.sample()
            predicted_id=predicted_id[None,:]
            
            if cached:
                seq_in=predicted_id # (1, 1) 
            else:
                seq_in=torch.concat([seq_in, predicted_id], axis=-1)
                
            output = torch.concat([output, predicted_id], axis=-1)

            if predicted_id[0,0] == end:
                break

        text = self.detokenize(output[0])

        if save_att is not None:
            logger.info("saving attention weights")
            cm.pklsave(save_att, att_weights)

        return text, att_weights, kvcachelst_nxt

    @torch.no_grad()
    def generate_text_infer(self, sentence, 
                      temperature=1.0, top_k=0, top_p=1.0, 
                      enable_kvcache=True,
                      max_length=50):
        '''
        Args:
            sentence: source sentence as input string.
            temperature: parameter to determine how deterministic the output is between (0,1]. 
                         Less deterministic on logits if temperature==1.
            top_k: value to select from top k largest logits, select all if k==0.
            top_p: cumulative probability threshold to select from logits for nucleus sampling. Select all if p == 1.
            enable_kvcache: enable caching of Key and Value tensors.
            max_length: maximum number of iterations to run.
        Returns:
            Predicted text, and max_length.
        '''
        assert 0.0 < temperature <=1.0, "set temperature between (0, 1]."
        assert 0.0 < top_p <=1.0, "set nucleus sampling probability between (0, 1], p=1 to skip top_p sampling."
        assert top_k >= 0, "set top_k above 0 or 0 to skip top_k sampling."

        sentence = torch.tensor(self.tokenize(sentence)).to(self.device)

        pldr_input = sentence
        end = torch.tensor(self.tokenize('')[0])
        output = pldr_input[:-1] if pldr_input.size()[0] > 1 else pldr_input # [seq_len]
        output = output[None, :] # [1, seq_len]
        seq_in=output
        kvcachelst=None
        cached=False

        for i in range(max_length):
            if not cached:
                combined_mask = self.create_masks(seq_in)
            else:
                combined_mask = None

            self.pldr_model.eval()
            predictions, _, kvcachelst_nxt  = self.pldr_model_infer([seq_in, combined_mask],
                                                           kvcachelst=kvcachelst)

            if enable_kvcache:
                kvcachelst=kvcachelst_nxt
                cached=True

            predictions = predictions[:, -1, :]  # (1, seq_len, vocab_size) -> (1, vocab_size)
            
            #temperature, top_k and nucleus sampling are stackable if needed.
            #scale logits for temperature sampling
            if temperature < 1:
                predictions = predictions/temperature

            #top_p sampling
            if top_p < 1:
                predictions=self.top_p_logits(logits=predictions, p=top_p)

            #top_k sampling
            if top_k > 0:
                predictions=self.top_k_logits(logits=predictions, k=top_k)
            

            predictions=torch.distributions.categorical.Categorical(logits=predictions) #(1, vocab_size)
            predicted_id=predictions.sample()
            predicted_id=predicted_id[None,:]
            
            if cached:
                seq_in=predicted_id # (1, 1)
            else:
                seq_in=torch.concat([seq_in, predicted_id], axis=-1)
                
            output = torch.concat([output, predicted_id], axis=-1)

            if predicted_id[0,0] == end:
                break

        text = self.detokenize(output[0])

        return text, kvcachelst_nxt



    @staticmethod
    def print_generated_text(sentence, full_output, max_eval_length):
        '''
        sentence: Input to the PLDR-LLM as string
        full_output: sentence+completion as output from PLDR-LLM as string
        max_eval_length: Number of maximum tokens for generation
        '''
        print(f"Max Eval Length: {max_eval_length}")
        print(f'{"Input":15s}: {sentence}')
        print(f'{"Prediction":15s}: {full_output}')


    @staticmethod
    @torch.no_grad()
    def accuracy_function(real, pred):
        accuracies = torch.eq(real, torch.argmax(pred, axis=2))

        mask = torch.ne(real, 0)
        accuracies = torch.logical_and(mask, accuracies)

        accuracies = accuracies.to(torch.float32)
        mask = mask.to(torch.float32)
        return torch.sum(accuracies) / torch.sum(mask)
    
def LinearWarmupCosineLRSchedule(optimizer, total_steps, warmup_steps, alpha=0.0, last_epoch=-1):
    total_steps=float(total_steps)
    warmup_steps=float(warmup_steps)
    alpha=float(alpha)
    def lr_schedule_fun(step):       
        step=float(step)
        step=min(step, total_steps)

        warmup_rise = (1/warmup_steps)*step

        decay_step=step-warmup_steps
        decay_rate = total_steps-warmup_steps
        cosine_decay = ((1-alpha)*0.5*(1+math.cos(math.pi*(decay_step/decay_rate)))+alpha)

        if step <=warmup_steps:
            return warmup_rise
        else:
            return cosine_decay
    
    return LambdaLR(optimizer, lr_schedule_fun, last_epoch)


def masked_loss_function(y_true, y_pred):
    mask=torch.ne(y_true,0)
    y_pred=torch.permute(y_pred, (0,2,1)) #[batch_size, seq_len, vocab_size]->[batch_size, vocab_size, seq_len]
    loss_=nn.CrossEntropyLoss(reduction='none')(y_pred, y_true)
    mask=mask.to(loss_.dtype)
    loss_*=mask
    loss= torch.sum(loss_)/torch.sum(mask)
    return loss

class AppState(Stateful):
    '''
    This is a wrapper that inherits the Stateful protocol that helps distributed checkpointing to load state
    dicts. 
    URL: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    '''

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        if self.optimizer is not None:
            model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer, 
                                                                    options=StateDictOptions(cpu_offload=True))
        else:
            model_state_dict = get_model_state_dict(self.model,
                                                    options=StateDictOptions(cpu_offload=True))
            optimizer_state_dict=None

        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        if self.optimizer is not None:
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"])
        else:
            set_model_state_dict(self.model, model_state_dict=state_dict["model"])
