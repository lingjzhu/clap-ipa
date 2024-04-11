import torch
import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Tuple

import webdataset as wds
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.core import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from transformers import DebertaV2Tokenizer, AutoProcessor
from sklearn.utils import shuffle

torch.set_num_threads(1)

    

class DataModule(LightningDataModule):
    def __init__(self, train_params, val_params, data_params):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params
        
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(data_params['tokenizer'])
        self.processor = AutoProcessor.from_pretrained(data_params['processor'])
        self.phoneset = [l.strip() for l in open(data_params['phoneset'],'r').readlines()]

    def _get_train_dataloader(self, cfg, train: bool) -> DataLoader:
        
        dataloader1 = self._get_wds_dataloader(cfg['fleurs'],train)
        
        dataloader2 = self._get_wds_dataloader(cfg['mswc'],train)
        
        iterables = {'mswc': dataloader2,
                    'fleurs': dataloader1}
        
        combined_loader = CombinedLoader(iterables, cfg['mode'])
        return combined_loader

    def train_dataloader(self) -> DataLoader:
        return self._get_train_dataloader(self.train_config, train=True)
    
    def val_dataloader(self) -> DataLoader:
        return self._get_wds_dataloader(self.val_config, train=False)
    
    def _get_wds_dataloader(self, cfg, train: bool) -> DataLoader:
        
        
        if train:
            filelist = shuffle([i.strip() for i in open(cfg['filelist_path'],'r').readlines()])
            dataset = (
                        wds.WebDataset(filelist)
                        .shuffle(3000)
                        .decode()
                        .to_tuple("npy", "txt", handler=wds.handlers.warn_and_continue)
                        )
            
            if cfg['use_hard_negatives']:
                collator = DataCollatorWithHardNegatives(processor=self.processor,
                                                         tokenizer=self.tokenizer,
                                                         max_length=cfg['max_length'],
                                                         phoneset = self.phoneset,
                                                         negative_prob = cfg['negative_prob'])
            else:
                collator = DataCollatorWithPadding(processor=self.processor,tokenizer=self.tokenizer,max_length=cfg['max_length'])
       
        else:
            filelist = [i.strip() for i in open(cfg['filelist_path'],'r').readlines()]
            dataset = (
                        wds.WebDataset(filelist)
                        .decode()
                        .to_tuple("npy", "txt", handler=wds.handlers.warn_and_continue)
                        )
            collator = DataCollatorWithPadding(processor=self.processor,tokenizer=self.tokenizer,max_length=cfg['max_length'])

        dataloader = DataLoader(dataset,num_workers=cfg['num_workers'],collate_fn=collator, batch_size=cfg['batch_size'])
        
        return dataloader
        
@dataclass
class DataCollatorWithPadding:
    processor: Any
    tokenizer: Any
    max_length: int = 320000
    max_label_length: int = 512

    def __call__(self, features: List[Tuple[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [feature[0] for feature in features]
        batch = self.processor(input_features, max_length=self.max_length, sampling_rate=16000, return_attention_mask=True,return_tensors="pt")
        
        length = torch.max(torch.sum(batch['attention_mask'],dim=-1))
        batch['attention_mask'] = batch['attention_mask'][:,:length]
        batch['input_features'] = batch['input_features'][:,:,:length]

        # get the tokenized label sequences
        phone_features = [feature[1] for feature in features]
        # pad the labels to max length
        phones_batch = self.tokenizer(phone_features, max_length=self.max_label_length,return_token_type_ids=False, truncation=True,padding=True, return_tensors="pt")
        #del phones_batch['token_type_ids']

        return batch, phones_batch


@dataclass
class DataCollatorWithHardNegatives:
    processor: Any
    tokenizer: Any
    max_length: int = 480000
    max_label_length: int = 512
    phoneset: List = None
    negative_prob: float = 0.1

    def __call__(self, features: List[Tuple[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [feature[0] for feature in features]
        batch = self.processor(input_features, max_length=self.max_length, sampling_rate=16000, return_attention_mask=True,return_tensors="pt")
        
        length = torch.max(torch.sum(batch['attention_mask'],dim=-1))
        batch['attention_mask'] = batch['attention_mask'][:,:length]
        batch['input_features'] = batch['input_features'][:,:,:length]

        # get the tokenized label sequences
        phone_features = [feature[1] for feature in features]
        # pad the labels to max length
        phones_batch = self.tokenizer(phone_features, max_length=self.max_label_length,truncation=True,padding=True,return_token_type_ids=False,  return_tensors="pt")
        #del phones_batch['token_type_ids']
        
        negatives = [generate_hard_negative(feature[1],self.phoneset, self.negative_prob) for feature in features]
        negatives_batch = self.tokenizer(negatives, max_length=self.max_label_length,truncation=True,padding=True,return_token_type_ids=False,  return_tensors="pt")
        #del negatives_batch['token_type_ids']

        return batch, phones_batch, negatives_batch
    

def augment(character, phoneset,no_del=False):
    action = np.random.uniform()
    if no_del:
        action = action*0.66
    if action <= 0.33: # replace
        return random.choice(phoneset)
    elif action > 0.33 and action <= 0.66: # insert
        return character + random.choice(phoneset) 
    elif action > 0.66: #delete
        return ' '
    
    
def generate_hard_negative(seq,phoneset,prob=0.3):
    if len(seq) == 1:
        negative = augment(seq,phoneset,no_del=True)
    else:
        if prob == 0:
            size = 1
        else:
            size = int(round(prob*len(seq)))
            size = max(size, 1)
        idx = np.random.uniform(size=size)
        idx = set([int(round(i*len(seq))) for i in idx])
        negative = ''.join([c if i not in idx else augment(c,phoneset) for i,c in enumerate(seq)]).replace(' ','')
    return negative