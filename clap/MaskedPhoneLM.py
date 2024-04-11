import re
import os
import json
import torch
import argparse
from dataclasses import dataclass, field
from transformers import DebertaV2Tokenizer
from transformers import BertForMaskedLM, BertConfig
from transformers import Trainer,TrainingArguments
from datasets import load_dataset
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import DataLoader

import wandb


def torch_mask_tokens(inputs, special_tokens_mask,mlm_probability=0.3):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    # We'll use the attention mask here
    special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    # 
    inputs[indices_replaced] = torch.tensor(tokenizer.mask_token_id)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels



@dataclass
class DataCollatorMPMWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: DebertaV2Tokenizer
    padding: Union[bool, str] = True
    return_attention_mask: Optional[bool] = True
    max_length_labels: Optional[int] = 514
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        label_features = [feature["input_ids"] for feature in features]


        batch = self.tokenizer(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                truncation=True,
                return_attention_mask=self.return_attention_mask,
                return_tensors="pt",
            )
        
        inputs, labels = torch_mask_tokens(batch['input_ids'],1-batch['attention_mask'])
        # replace padding with -100 to ignore loss correctly
        labels = labels.masked_fill(batch.attention_mask.ne(1), -100)
        
        batch['input_ids'] = inputs
        batch["labels"] = labels

        return batch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir', type=str, default='/home/lingjzhu/scratch/cache')
    parser.add_argument('--out_dir',type=str,default="/home/lingjzhu/scratch/PhoneBERT")
    parser.add_argument('--tokenizer',type=str,default='./tokenizers/unigram')
    parser.add_argument('--name',type=str,default='bert-base-uncased')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--grad_acc',type=int,default=1)
    args = parser.parse_args()

    run = wandb.init(
                # Set the project where this run will be logged
                project="MaskedPhoneLM",
                name=args.name
                )
    # load text dataset
    corpus = load_dataset('anyspeech/PhoneCorpus',cache_dir=args.cache_dir,keep_in_memory=True)['train']
    corpus = corpus.rename_column('phones','input_ids')

    tokenizer = DebertaV2Tokenizer.from_pretrained(args.tokenizer)

    # load model
    if args.name == 'bert-base-uncased' or args.name == 'bert-large-uncased':
        config = BertConfig.from_pretrained(args.name)
    elif args.name == 'bert-small':
        config = BertConfig()
        config.hidden_size = 512
        config.num_hidden_layers = 6
        config.num_attention_heads = 8
        config.intermediate_size = 2048
    elif args.name == 'bert-tiny':
        config = BertConfig()
        config.hidden_size = 384
        config.num_hidden_layers = 4
        config.num_attention_heads = 6
        config.intermediate_size = 1536

        
    config.max_position_embeddings = 514
    config.vocab_size = len(tokenizer)
    model = BertForMaskedLM(config)

    data_collator = DataCollatorMPMWithPadding(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
                                      output_dir=os.path.join(args.out_dir,args.name),
                                      group_by_length=False,
                                      per_device_train_batch_size=args.batch_size,
                                      gradient_accumulation_steps=args.grad_acc,
                                      num_train_epochs=3,
                                      fp16=True,
                                      gradient_checkpointing=True,
                                      save_steps=5000,
                                      logging_steps=50,
                                      learning_rate=1e-4,
                                      weight_decay=0.001,
                                      warmup_steps=1000,
                                      save_total_limit=2,
                                      ignore_data_skip=True,
                                      report_to="wandb",
                                      dataloader_num_workers=8,
                                     )
        
        
    trainer = Trainer(
                        model=model,
                        data_collator=data_collator,
                        args=training_args,
                        train_dataset=corpus,
 #                       eval_dataset=books['val'], 
                     )
    
    
    trainer.train()


