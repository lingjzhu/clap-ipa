import torch
import numpy as np
import random
import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Tuple

import webdataset as wds
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.core import LightningDataModule
from lightning.pytorch.utilities import CombinedLoader
from transformers import DebertaV2Tokenizer, AutoProcessor, AutoTokenizer
from sklearn.utils import shuffle
from lingpy.sequence.sound_classes import ipa2tokens


torch.set_num_threads(1)

    

class DataModule(LightningDataModule):
    def __init__(self, train_params, val_params, data_params):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params
        
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(data_params['tokenizer'])
        self.tokenizer.bos_token_id = self.tokenizer.eos_token_id
        self.processor = AutoProcessor.from_pretrained(data_params['processor'])
        self.phoneset = [l.strip() for l in open(data_params['phoneset'],'r').readlines()]

    def train_dataloader(self) -> DataLoader:
        return self._get_wds_dataloader(self.train_config, train=True)
    
    def val_dataloader(self) -> DataLoader:
        return self._get_wds_dataloader(self.val_config, train=False)
    
    def _get_wds_dataloader(self, cfg, train: bool) -> DataLoader:
        
        
        if train:
            filelist = shuffle([i.strip() for i in open(cfg['filelist_path'],'r').readlines()])
            dataset = (
                        wds.WebDataset(filelist)
                        .shuffle(1000)
                        .decode()
                        .to_tuple("npy", "txt", handler=wds.handlers.warn_and_continue)
                        )
            
            collator = DataCollatorWithPadding(processor=self.processor,
                                               tokenizer=self.tokenizer,
                                               max_length=cfg['max_length'],
                                               phoneset = self.phoneset,
                                               negative_prob = cfg['negative_prob'])
       
        else:
            filelist = [i.strip() for i in open(cfg['filelist_path'],'r').readlines()]
            dataset = (
                        wds.WebDataset(filelist)
                        .decode()
                        .to_tuple("npy", "txt", handler=wds.handlers.warn_and_continue)
                        )
            collator = DataCollatorWithPadding(processor=self.processor,
                                               tokenizer=self.tokenizer,
                                               max_length=cfg['max_length'],
                                               phoneset = self.phoneset,
                                               negative_prob = cfg['negative_prob'])

        dataloader = DataLoader(dataset,num_workers=cfg['num_workers'],collate_fn=collator, batch_size=cfg['batch_size'])
        
        return dataloader
        
@dataclass
class DataCollatorWithPadding:
    processor: Any
    tokenizer: Any
    max_length: int = 320000
    max_label_length: int = 512
    phoneset: List = None
    negative_prob: float = 0.1
    #resolutions = [(2,1),(5,3),(10,5)]

    def __call__(self, features: List[Tuple[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        # get the tokenized label sequences
        '''
        prob = np.random.uniform()
        if prob <= 0.5:
            win_len=1
            shift=1
            phone_chunking = False
        else:
            (win_len, shift) = self.resolutions[np.random.randint(3)]
            phone_chunking = True
        '''
        input_features = [feature[0] for feature in features]
        batch = self.processor(input_features, max_length=self.max_length, sampling_rate=16000, return_attention_mask=True,return_tensors="pt")
        
        length = torch.max(torch.sum(batch['attention_mask'],dim=-1))
        batch['attention_mask'] = batch['attention_mask'][:,:length]
        batch['input_features'] = batch['input_features'][:,:,:length]

        speech_lengths = torch.sum(batch['attention_mask'][:,::2],dim=-1)


        phone_features = [ipa2tokens(feature[1].replace(' ',''),merge_vowels=False) for feature in features]
        
        # pad the labels to max length
        phones_batch = []
        phone_masks = []
        phone_lengths = []
        for phone_feature in phone_features:
            out = self.tokenizer(phone_feature, return_attention_mask=False, return_length=True, return_token_type_ids=False, add_special_tokens=False)
            phones = [self.tokenizer.eos_token_id] + list(itertools.chain.from_iterable(out['input_ids'])) + [self.tokenizer.eos_token_id]
            #  eos id represents silence
            lengths = [1] + out['length'] + [1]
            mask = self._create_phone_mask(lengths)
            phones_batch.append({'input_ids':phones})
            phone_masks.append(mask[:,:self.max_label_length])
            phone_lengths.append(len(lengths))

        phones_batch = self.tokenizer.pad(phones_batch, 
                                          max_length=self.max_label_length, 
                                          padding='longest',
                                          return_attention_mask=True,
                                          return_tensors="pt")
        # somehow the above function cannot truncate inputs, so do it manually
        phones_batch['input_ids'] = phones_batch['input_ids'][:,:self.max_label_length]
        phones_batch['attention_mask'] = phones_batch['attention_mask'][:,:self.max_label_length]
        
        phone_transform_masks = torch.zeros(phones_batch['attention_mask'].size(0),
                                            max(phone_lengths),
                                            phones_batch['attention_mask'].size(1))
        for i,mask in enumerate(phone_masks):
            phone_transform_masks[i, :mask.size(0),:mask.size(1)] = mask
            
            
        ctc_targets = self.tokenizer([feature[1].replace(' ','') for feature in features],return_attention_mask=False,return_length=True, return_token_type_ids=False,max_length=512,truncation=True, padding=True,return_tensors='pt')
        

        return batch, phones_batch, speech_lengths, torch.tensor(phone_lengths).long(), phone_transform_masks, ctc_targets
    
    def _create_phone_mask(self,lengths):

        mask = torch.zeros(len(lengths),sum(lengths))

        cumsum = 0
        for i,l in enumerate(lengths):
            mask[i,cumsum:cumsum+l] = 1/l+1e-8
            cumsum += l

        return mask
    
    def _create_sliding_window(self,lengths,win_len=1,shift=1):

        B = lengths.size(0)
        L = torch.max(lengths)
        num_win = torch.ceil(L/shift).long()
        sliding_window = torch.zeros(B,num_win,L)
        for n in range(num_win):
            sliding_window[:,n,n*shift:min(n*shift+win_len,L)] = 1.0
        #sliding_window = sliding_window/torch.sum(sliding_window,dim=1).unsqueeze(1)
        for i,l in enumerate(lengths):
            sliding_window[i,:,l:] = 0.0
            sliding_window[i] *= 1/(torch.sum(sliding_window[i],dim=-1).unsqueeze(-1)+1e-8)
        transform_lengths = sliding_window.sum(dim=-1).sum(dim=-1).long()
        return sliding_window, transform_lengths

    def _create_phone_chunks(self,seq):

        max_step = len(seq)//5
        step = 5#np.random.randint(low=2,high=10)
        step = np.minimum(max_step,step)
        chunks = [''.join(seq[i:i+step]) for i in range(0,len(seq),step)]

        return chunks
