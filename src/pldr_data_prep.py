'''
Data preparation module for Large Language Model from Power Law Decoder Representations.
Loads a dataset ready for pretraining. Presets for falcon-refinedweb are available.
'''

import logging
logging.basicConfig(format='%(levelname)s:%(name)s: %(message)s')
logger=logging.getLogger("D")
logger.setLevel(logging.DEBUG)

import datasets as hfds
from datasets.distributed import split_dataset_by_node
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

import torch
import torch.nn as nn
from torchtune.modules.tokenizers import SentencePieceBaseTokenizer

class dist_pldr_data_prep:
    '''
    Prepares dataset for Large Language Model from Power Law Decoder Representations (PLDR-LLM) for pretraining
    Optimized for loading falcon-refinedweb dataset
    '''
    def __init__(self,
                 rank,
                 WORLD_SIZE,
                 BUFFER_SIZE=20000,
                 BATCH_SIZE = 32,
                 dataset_file="tiiuae/falcon-refinedweb",
                 dataset_column_label="content",
                 split_names=None,
                 load_dataset=True,
                 load_from_train=False, 
                 split_style="index",
                 train_intvl=None, 
                 val_offset=None,
                 test_offset=None, 
                 tok_model = "/path/to/tokenizer/model",
                 shuffle_set=False, 
                 MAX_LENGTH=None,
                 batch_agg_count=None, 
                 padding_type='pack',
                 trust_remote_code=False,
                 ):
        '''
        Args:
            BUFFER_SIZE: Buffer size for shuffling
            BATCH_SIZE: Batch size for dataset
            dataset_file: path to huggingface/tensorflow dataset
            split_names: Names for split if they are different from [train, validation, test].
            load_dataset: if True load the dataset
            load_from_train: load from train split only
            split_style: 'percent' or 'index' based splitting of dataset
            train_intvl: input is a tuple integer (start,end), None loads all data. 
            val_offset: input is single integer as offset, None skips validation dataset
            test_offset: input is single integer as offset, None skips test dataset.
            tok_model: file path for tokenizer model
            shuffle_set: If True, shuffle the dataset while loading
            MAX_LENGTH: maximum number of tokens in each sample.
            batch_agg_count: the multiplier for batch_size to densify batches by concatenating
            padding_type: type of completing partially filled sample up to MAX_LENGTH. pack completes with more tokens
                          pad adds padding, nopad skips padding.
            trust_remote_code: Set this to true to enable remote code while downloading dataset.

        Returns batched, tokenized train, validation and/or test datasets ready for pretraining. 
        Tokenizer methods are accessible through instance of this class object
        '''
        self.rank=rank
        self.WORLD_SIZE=WORLD_SIZE
        self.BATCH_SIZE=BATCH_SIZE
        self.BUFFER_SIZE=BUFFER_SIZE
        self.MAX_LENGTH = MAX_LENGTH if MAX_LENGTH is not None else 512
        self.AGG_BATCHES=batch_agg_count if batch_agg_count is not None else 100
        self.tok_model = tok_model
        self.tokenizer=self.load_sentencepiece_tokenizer()
        self.col_name=dataset_column_label
        self.short_chunk=None
        self.padding_type=padding_type
        self.trust_remote_code=trust_remote_code

        #load dataset
        if split_names is not None:
            split_train, split_val, split_test=split_names
        else:
            split_train='train'
            split_val='validation'
            split_test='test'
        
        self.val_examples=None
        self.test_examples=None

        if load_dataset:
            logger.info(f"(Rank {self.rank}) LOADING DATASET")
            if load_from_train:
                #Load dataset from train data
                if train_intvl:
                    start_ind, end_ind=train_intvl
                    
                    assert (val_offset is None) or val_offset > 0, "WARNING: validation offset should be positive"
                    assert (test_offset is None) or (test_offset > 0 and test_offset > end_ind+val_offset), \
                                                                    "WARNING: test offset is overlapping validation data"
                    #load only percentage of train data
                    if  not val_offset and not test_offset:
                        if split_style=='percent':
                            examples = hfds.load_dataset(dataset_file, split=[f'{split_train}[{start_ind}%:{end_ind}%]'], 
                                                         trust_remote_code=self.trust_remote_code)
                        elif split_style=='index':
                            examples = hfds.load_dataset(dataset_file, split=[f'{split_train}[{start_ind}:{end_ind}]'],
                                                        trust_remote_code=self.trust_remote_code)
                        logger.info(f"(Rank {self.rank}) Dataset Info for {dataset_file}:")
                        logger.info(f"(Rank {self.rank}) Train: {examples[0]}")
                        self.train_examples=examples[0].select_columns([self.col_name])
                    #load only percentage of train data and rest for validation set
                    elif val_offset and not test_offset:
                        if split_style=='percent':
                            examples = hfds.load_dataset(dataset_file,
                                                         split=[f'{split_train}[{start_ind}%:{end_ind}%]', 
                                                                f'{split_train}[{end_ind}%:{end_ind+val_offset}%]'], 
                                                        trust_remote_code=self.trust_remote_code)
                        elif split_style=='index':
                            examples = hfds.load_dataset(dataset_file,
                                                        split=[f'{split_train}[{start_ind}:{end_ind}]', 
                                                                f'{split_train}[{end_ind}:{end_ind+val_offset}]'], 
                                                                trust_remote_code=self.trust_remote_code)

                        logger.info(f"(Rank {self.rank}) Dataset Info for {dataset_file}:")
                        logger.info(f"(Rank {self.rank}) Train: {examples[0]}")
                        logger.info(f"(Rank {self.rank}) Validation: {examples[1]}")
                        self.train_examples =examples[0].select_columns([self.col_name])
                        self.val_examples=examples[1].select_columns([self.col_name])
                    elif test_offset:
                        if split_style=='percent':
                            examples = hfds.load_dataset(dataset_file,
                                                        split=[f'{split_train}[{start_ind}%:{end_ind}%]', 
                                                                f'{split_train}[{end_ind}%:{end_ind+val_offset}%]',
                                                                f'{split_train}[{end_ind+val_offset}%:{end_ind+val_offset+test_offset}%]'], 
                                                        trust_remote_code=self.trust_remote_code)
                        elif split_style=='index':
                            examples = hfds.load_dataset(dataset_file,
                                                        split=[f'{split_train}[{start_ind}:{end_ind}]', 
                                                                f'{split_train}[{end_ind}:{end_ind+val_offset}]',
                                                                f'{split_train}[{end_ind+val_offset}:{end_ind+val_offset+test_offset}]'], 
                                                        trust_remote_code=self.trust_remote_code)
                        logger.info(f"(Rank {self.rank}) Dataset Info for {dataset_file}:")
                        logger.info(f"(Rank {self.rank}) Train: {examples[0]}")
                        logger.info(f"(Rank {self.rank}) Validation: {examples[1]}")
                        logger.info(f"(Rank {self.rank}) Test: {examples[2]}")
                        self.train_examples=examples[0].select_columns([self.col_name]) 
                        self.val_examples=examples[1].select_columns([self.col_name]) 
                        self.test_examples=examples[2].select_columns([self.col_name])                 
                else:
                    examples = hfds.load_dataset(dataset_file,
                                                 split=[f'{split_train}'], 
                                                 trust_remote_code=self.trust_remote_code)

                    logger.info(f"(Rank {self.rank}) Dataset Info for {dataset_file}:")
                    logger.info(f"(Rank {self.rank}) Train: {examples[0]}")
                    self.train_examples=examples[0].select_columns([self.col_name])                   
            else:
                #Load dataset for train, validation and test
                if train_intvl:
                    start_ind, end_ind=train_intvl
                    if split_style=='percent':
                        examples = hfds.load_dataset(dataset_file,
                                                        split=[f'{split_train}[{start_ind}%:{end_ind}%]', split_val, split_test], 
                                                        trust_remote_code=self.trust_remote_code)
                    elif split_style=='index':
                        examples = hfds.load_dataset(dataset_file,
                                                     split=[f'{split_train}[{start_ind}:{end_ind}]', split_val, split_test], 
                                                     trust_remote_code=self.trust_remote_code)
                else:
                    examples = hfds.load_dataset(dataset_file,
                                                 split=[split_train, split_val, split_test], 
                                                 trust_remote_code=self.trust_remote_code)

                logger.info(f"(Rank {self.rank}) Dataset Info for {dataset_file}:")
                logger.info(f"(Rank {self.rank}) Train: {examples[0]}")
                logger.info(f"(Rank {self.rank}) Validation: {examples[1]}")
                logger.info(f"(Rank {self.rank}) Test: {examples[2]}")

                self.train_examples=examples[0].select_columns([self.col_name])
                self.val_examples=examples[1].select_columns([self.col_name]) 
                self.test_examples=examples[2].select_columns([self.col_name])

            logger.info(f"(Rank {self.rank}) BEGINNING PREPROCESSING EXAMPLES FOR {dataset_file}")
            self.tokenize_fun=self.tokenize_hfds
            self.detokenize_fun=self.detokenize_hfds

            #rename column to a fixed value for easy reference.
            self.train_examples=self.train_examples.rename_column(self.col_name, 'sample')
            if self.val_examples:
                self.val_examples=self.val_examples.rename_column(self.col_name, 'sample')
            if self.test_examples:
                self.test_examples=self.test_examples.rename_column(self.col_name, 'sample')
            self.col_name='sample'
            
            logger.info(f"(Rank {self.rank}) SHARDING DATASETS AMONG RANKS")
            logger.info(f"(Rank {self.rank}) CREATING BATCHED DATASETS FOR TRAINING AND VALIDATION")
            self.train_batches=self.split_and_make_batches(self.train_examples, shuffle_set=shuffle_set, num_shards=64)
            if self.val_examples:
                self.val_batches=self.split_and_make_batches(self.val_examples, shuffle_set=False, num_shards=64)
            if self.test_examples:
                self.test_batches=self.split_and_make_batches(self.test_examples, shuffle_set=False, num_shards=64)
            logger.info(f"(Rank {self.rank}) DONE PREPROCESSING EXAMPLES FOR {dataset_file}")   
            logger.info(f"(Rank {self.rank}) BATCHED DATASETS ARE CREATED")
    
        else:
            logger.info(f"(Rank {self.rank}) SKIPPED LOADING DATASET")

    def split_and_make_batches(self, examples, shuffle_set=False, num_shards=64):
        examples=split_dataset_by_node(examples, rank=self.rank, world_size=self.WORLD_SIZE)
        examples=examples.to_iterable_dataset(num_shards=num_shards)
        logger.info(f"(Rank {self.rank}) Dataset Info for Train Examples on Rank:")
        logger.info(f"(Rank {self.rank}) Train: {examples}")
        batches=self.make_dense_padded_batches(examples, shuffle_set=shuffle_set)
        return batches


    def tokenize_hfds(self, input):
        '''
        For input as dict and output as tensor
        '''
        input[self.col_name]=self.tokenizer.encode(input[self.col_name], add_bos=False, add_eos=True, 
                                     trim_leading_whitespace=False, prefix=None)
        input[self.col_name]=torch.tensor(input[self.col_name])
        return input

    def detokenize_hfds(self, input):
        '''
        For input as dict and output as tensor
        '''        
        input[self.col_name]=self.tokenizer.decode(input[self.col_name])
        input[self.col_name]=torch.tensor(input[self.col_name])
        return input
    
    def tokenize(self, input):
        '''
        For input as tensor
        '''
        return self.tokenizer.encode(input, add_bos=False, add_eos=True, 
                                     trim_leading_whitespace=False, prefix=None)

    def detokenize(self, input):
        '''
        For input as tensor
        '''
        return self.tokenizer.decode(input.tolist())

    def load_sentencepiece_tokenizer(self):
        tokenizer=SentencePieceBaseTokenizer(self.tok_model)
        logger.info(f"(Rank {self.rank}) Tokenizer items:")
        logger.info([item for item in dir(tokenizer) if not item.startswith('_')])
        logger.info(f"(Rank {self.rank}) Tokenizer Loaded.")
        return tokenizer

    def concat_chunk_batches_packed(self, input):
        '''
        input is a dict of tokenized samples with a large batch size.
        This method concatenates them, creates chunks of context length.
        The short chunk tracks the last chunk with shorter than context length,
        and appends it to the beginning of next input. This removes the need for
        padding except for very last chunk, which may be dropped.
        '''
        
        if self.short_chunk is not None:
            #if there is left over from previous chunk, add it to next big batch in the front.
            input[self.col_name].insert(0, self.short_chunk)

        input[self.col_name]=torch.concat(input[self.col_name], dim=-1)
        input[self.col_name]=torch.split(input[self.col_name], split_size_or_sections=self.MAX_LENGTH, dim=-1)
        input[self.col_name]=list(input[self.col_name])

        #Update short chunk with the new remainder token ids
        self.short_chunk=input[self.col_name][-1] if input[self.col_name][-1].size()[0] < self.MAX_LENGTH else None
        input[self.col_name]=input[self.col_name][:-1] if self.short_chunk is not None else input[self.col_name]

        return input
    
    def concat_chunk_batches_pad(self, input):
        '''
        input is a dict of tokenized samples with a large batch size.
        This method concatenates them, creates chunks of context length.
        The last partially filled chunk is padded to context length.
        '''

        input[self.col_name]=torch.concat(input[self.col_name], dim=-1)
        input[self.col_name]=torch.split(input[self.col_name], split_size_or_sections=self.MAX_LENGTH, dim=-1)
        input[self.col_name]=list(input[self.col_name])
        input[self.col_name]=nn.utils.rnn.pad_sequence(input[self.col_name], batch_first=True, 
                                         padding_value=0, padding_side="right")

        return input
    
    def concat_chunk_batches_nopad(self, input):
        '''
        input is a dict of tokenized samples with a large batch size.
        This method concatenates them, creates chunks of context lengt.
        The last partially filled chunk is left without padding.
        '''

        input[self.col_name]=torch.concat(input[self.col_name], dim=-1)
        input[self.col_name]=torch.split(input[self.col_name], split_size_or_sections=self.MAX_LENGTH, dim=-1)
        input[self.col_name]=list(input[self.col_name])

        return input

    def make_dense_padded_batches(self, ds, shuffle_set=False):
        ''' To be used with pretraining train and val dataset'''
        logger.info(f"(Rank {self.rank}) Tokenizing samples.")
        ds=ds.map(function=self.tokenize_fun, batched=False)
        if shuffle_set:
            logger.info(f"(Rank {self.rank}) Shuffling Dataset after Tokenization")
            ds=ds.shuffle(writer_batch_size=self.BUFFER_SIZE)
        
        logger.info(f"(Rank {self.rank}) Batching by concatenating and chunking samples with padding style: {self.padding_type}.")
        
        if self.padding_type=='pad':
            ds = ds.map(self.concat_chunk_batches_pad, batched=True,
                        batch_size=int(self.BATCH_SIZE*self.AGG_BATCHES),
                        )
        elif self.padding_type=='pack':
            ds = ds.map(self.concat_chunk_batches_packed, batched=True,
                        batch_size=int(self.BATCH_SIZE*self.AGG_BATCHES),
                        )
        elif self.padding_type=='nopad':
                        ds = ds.map(self.concat_chunk_batches_nopad, batched=True,
                                    batch_size=int(self.BATCH_SIZE*self.AGG_BATCHES)
                                    )
        else:
            logger.info(f"(Rank {self.rank}) Please choose from available options for padding_type: 'pack', 'pad', or 'nopad'")
        
        ds_batched=torch.utils.data.DataLoader(ds, batch_size=self.BATCH_SIZE, shuffle=False,
                                                drop_last=True, num_workers=0, 
                                                prefetch_factor=None)
        
        logger.info(f"(Rank {self.rank}) DATA PREPARATION IS DONE.")

        return ds_batched