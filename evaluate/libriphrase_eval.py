import os
import numpy as np
import soundfile as sf
import torch
from transformers import BertModel,DebertaV2Tokenizer, AutoProcessor, AutoTokenizer
import clap.encoders
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Tuple
import pandas as pd
import sklearn.metrics
import wandb

def load_audio(path):
    y,sr = sf.read(path)
    return y


@dataclass
class DataCollatorWithPadding:
    processor: Any
    tokenizer: Any
    max_length: int = 80000
    max_label_length: int = 256
    text: bool = False

    def __call__(self, features):
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        anchor_features = [load_audio(feature['anchor']) for feature in features]
        anchor_batch = self.processor(anchor_features, max_length=self.max_length, sampling_rate=16000, return_attention_mask=True,return_tensors="pt")
        
        anchor_length = torch.max(torch.sum(anchor_batch['attention_mask'],dim=-1))
        anchor_batch['attention_mask'] = anchor_batch['attention_mask'][:,:anchor_length]
        anchor_batch['input_features'] = anchor_batch['input_features'][:,:,:anchor_length]
        
        comparison_features = [load_audio(feature['comparison']) for feature in features]
        comparison_batch = self.processor(comparison_features, max_length=self.max_length, sampling_rate=16000, return_attention_mask=True,return_tensors="pt")
        
        comparison_length = torch.max(torch.sum(comparison_batch['attention_mask'],dim=-1))
        comparison_batch['attention_mask'] = comparison_batch['attention_mask'][:,:comparison_length]
        comparison_batch['input_features'] = comparison_batch['input_features'][:,:,:comparison_length]

        # get the tokenized label sequences
        if self.text:
            anchor_phone_features = [feature['anchor_text'] for feature in features]
        else:
            anchor_phone_features = [feature['anchor_phone'] for feature in features]
        # pad the labels to max length
        anchor_phones_batch = self.tokenizer(anchor_phone_features, max_length=self.max_label_length,truncation=True,padding=True, 
                                             return_token_type_ids=False,return_tensors="pt")
        
         # get the tokenized label sequences
        if self.text:
            comparison_phone_features = [feature['comparison_text'] for feature in features]
        else:
            comparison_phone_features = [feature['comparison_phone'] for feature in features]
        # pad the labels to max length
        comparison_phones_batch = self.tokenizer(comparison_phone_features, max_length=self.max_label_length,truncation=True,padding=True, 
                                             return_token_type_ids=False,return_tensors="pt")       
 
        
        types = [feature['type'] for feature in features]
        labels = [feature['target'] for feature in features]
        return anchor_batch, comparison_batch, anchor_phones_batch, comparison_phones_batch, types, labels


def load_models(checkpoint_path):

    checkpoints = torch.load(checkpoint_path)
    
    speech_encoder = clap.encoders.initialize_speech_model(checkpoints['hyper_parameters']['speech_encoder_cfg'])
    phone_encoder = clap.encoders.initialize_phone_model(checkpoints['hyper_parameters']['phone_encoder_cfg'])

    speech_encoder.load_state_dict({k.replace('speech_encoder.',''):v for k,v in checkpoints['state_dict'].items() if 'speech_encoder' in k},strict=True)
    phone_encoder.load_state_dict({k.replace('phone_encoder.',''):v for k,v in checkpoints['state_dict'].items() if 'phone_encoder' in k},strict=True)

    return phone_encoder.to('cuda').eval(), speech_encoder.to('cuda').eval()


def map_path(batch):
    batch['anchor'] = os.path.join(basepath,batch['anchor'])
    batch['comparison'] = os.path.join(basepath,batch['comparison'])
    return batch



def compute_eer(label, pred, positive_label=1):
    '''
    https://github.com/YuanGongND/python-compute-eer
    '''
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    return eer




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--basepath', type=str, default="/home/lingjzhu/scratch/libriphrase")
    parser.add_argument('--tokenizer',type=str,default='charsiu/IPATokenizer')
    parser.add_argument('--dataset',type=str, default='charsiu/libriphrase_meta')
    parser.add_argument('--checkpoint',type=str)
    parser.add_argument('--pooling',type=str,default='mean')
    parser.add_argument('--name',type=str)
    parser.add_argument('--cache_dir',type=str,default='/home/lingjzhu/scratch/cache')
    parser.add_argument('--text',action='store_true')
    args = parser.parse_args()
    
    
    run = wandb.init(
                # Set the project where this run will be logged
                project="clap-ipa-eval",
                name=args.name
                )
    
    basepath = args.basepath
    
    outpath = os.path.join(args.basepath,args.name+'.csv')
    
    processor = AutoProcessor.from_pretrained('openai/whisper-base')
    if args.text:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.tokenizer)
    
    dataset = load_dataset(args.dataset,split="train", cache_dir=args.cache_dir)
    dataset = dataset.map(map_path)
    
    phone_encoder, speech_encoder = load_models(args.checkpoint)
    
    collator = DataCollatorWithPadding(processor=processor,tokenizer=tokenizer,text=args.text)
    dataloader = DataLoader(dataset, batch_size=512, collate_fn=collator, num_workers=8)
    device = 'cuda' 
    
    with open(outpath,'w') as out:
        out.write('%s,%s,%s,%s,%s\n'%('s2s_score', 'p2s_score', 's2p_score', 'type', 'label'))
        for anchor_batch, comparison_batch, anchor_phones_batch, comparison_phones_batch, types, labels in tqdm(dataloader):
            with torch.no_grad():
                z_anchor = speech_encoder(**anchor_batch.to(device)).pooler_output
                z_comparison = speech_encoder(**comparison_batch.to(device)).pooler_output

                z_anchor_phones = phone_encoder(**anchor_phones_batch.to(device)).pooler_output
                z_comparison_phones = phone_encoder(**comparison_phones_batch.to(device)).pooler_output
                    
                s2s_scores = list(F.cosine_similarity(z_anchor,z_comparison).cpu().numpy())
                p2s_scores = list(F.cosine_similarity(z_anchor_phones,z_comparison).cpu().numpy())
                s2p_scores = list(F.cosine_similarity(z_anchor,z_comparison_phones).cpu().numpy())

                
                for s2s, p2s, s2p, t, l in zip(s2s_scores, p2s_scores,s2p_scores, types, labels):
                    out.write('%s,%s,%s,%s,%s\n'%(s2s, p2s, s2p, t, l))
        
    data = pd.read_csv(outpath)
    easy = data[data['type']!='diffspk_hardneg']
    hard = data[data['type']!='diffspk_easyneg']
    
    
    wandb.log({'libriphrase/libri_easy_s2s_auc': sklearn.metrics.roc_auc_score(easy['label'],easy['s2s_score']),
                'libriphrase/libri_easy_p2s_auc': sklearn.metrics.roc_auc_score(easy['label'],easy['p2s_score']),
                'libriphrase/libri_easy_s2p_auc': sklearn.metrics.roc_auc_score(easy['label'],easy['s2p_score']),
                'libriphrase/libri_hard_s2s_auc': sklearn.metrics.roc_auc_score(hard['label'],hard['s2s_score']),
                'libriphrase/libri_hard_p2s_auc': sklearn.metrics.roc_auc_score(hard['label'],hard['p2s_score']),
                'libriphrase/libri_hard_s2p_auc': sklearn.metrics.roc_auc_score(hard['label'],hard['s2p_score'])}
             )
    
    wandb.log({'libriphrase/libri_easy_s2s_eer': compute_eer(easy['label'],easy['s2s_score']),
                'libriphrase/libri_easy_p2s_eer': compute_eer(easy['label'],easy['p2s_score']),
                'libriphrase/libri_easy_s2p_eer': compute_eer(easy['label'],easy['s2p_score']),
                'libriphrase/libri_hard_s2s_eer': compute_eer(hard['label'],hard['s2s_score']),
                'libriphrase/libri_hard_p2s_eer': compute_eer(hard['label'],hard['p2s_score']),
                'libriphrase/libri_hard_s2p_eer': compute_eer(hard['label'],hard['s2p_score'])}
            )