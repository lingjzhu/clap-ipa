import itertools
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm 
from datasets import load_dataset
from lingpy.sequence.sound_classes import ipa2tokens
from librosa.sequence import dtw
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, DebertaV2Tokenizer

import clap
from clap.encoders import *


import argparse
import wandb

def evaluate_overlap(evaluation_pairs):
    
    hits = 0
    counts = 0
    for targets,preds in tqdm(evaluation_pairs):
        assert len(targets)==len(preds)
        hits += sum(np.array(targets)==np.array(preds))
        counts += len(targets)
        
    return hits/counts


'''
Code for precision, recall, F1, R-value was adapted from unsupseg: https://github.com/felixkreuk/UnsupSeg
'''

def get_metrics(precision_counter, recall_counter, pred_counter, gt_counter):
    eps = 1e-7

    precision = precision_counter / (pred_counter + eps)
    recall = recall_counter / (gt_counter + eps)
    f1 = 2 * (precision * recall) / (precision + recall + eps)

    os = recall / (precision + eps) - 1
    r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
    r2 = (-os + recall - 1) / (np.sqrt(2))
    rval = 1 - (np.abs(r1) + np.abs(r2)) / 2

    return precision, recall, f1, rval


def get_stats(y, y_ids, yhat, yhat_ids, tolerance=0.02):

    precision_counter = 0
    recall_counter = 0
    pred_counter = 0
    gt_counter = 0

    for yhat_i, yhat_id in zip(yhat,yhat_ids):
        diff = np.abs(y - yhat_i)
        min_dist = diff.min()
        min_pos = np.argmin(diff)
        intersect = set(y_ids[min_pos]).intersection(set(yhat_id))
        if len(intersect)>0:
            precision_counter += (min_dist <= tolerance)

    for y_i,y_id in zip(y,y_ids):
        diff = np.abs(yhat - y_i)
        min_dist = diff.min()
        min_pos = np.argmin(diff)
        intersect = set(yhat_ids[min_pos]).intersection(set(y_id))
        if len(intersect)>0:
            recall_counter += (min_dist <= tolerance)

    pred_counter += len(yhat)
    gt_counter += len(y)

    p, r, f1, rval = get_metrics(precision_counter,
                                      recall_counter,
                                      pred_counter,
                                      gt_counter)
    return p, r, f1, rval

def get_all_stats(evaluation_pairs,tolerance=0.1):
    
    precision_counter = 0
    recall_counter = 0
    pred_counter = 0
    gt_counter = 0
    
    for (y,y_ids), (yhat, yhat_ids) in tqdm(evaluation_pairs):
        
        for yhat_i, yhat_id in zip(yhat,yhat_ids):
            diff = np.abs(y - yhat_i)
            min_dist = diff.min()
            min_pos = np.argmin(diff)
            intersect = set(y_ids[min_pos]).intersection(set(yhat_id))
            if len(intersect)>0:
                precision_counter += (min_dist <= tolerance)

        for y_i,y_id in zip(y,y_ids):
            diff = np.abs(yhat - y_i)
            min_dist = diff.min()
            min_pos = np.argmin(diff)
            intersect = set(yhat_ids[min_pos]).intersection(set(y_id))
            if len(intersect)>0:
                recall_counter += (min_dist <= tolerance)

        pred_counter += len(yhat)
        gt_counter += len(y)
        
    p, r, f1, rval = get_metrics(precision_counter,
                                      recall_counter,
                                      pred_counter,
                                      gt_counter)
    return p, r, f1, rval
 


def get_all_stats_boundary_only(evaluation_pairs,tolerance=0.1):
    
    precision_counter = 0
    recall_counter = 0
    pred_counter = 0
    gt_counter = 0
    
    for (y,y_ids), (yhat, yhat_ids) in tqdm(evaluation_pairs):
        
        for yhat_i, yhat_id in zip(yhat,yhat_ids):
            diff = np.abs(y - yhat_i)
            min_dist = diff.min()
            min_pos = np.argmin(diff)
            precision_counter += (min_dist <= tolerance)

        for y_i,y_id in zip(y,y_ids):
            diff = np.abs(yhat - y_i)
            min_dist = diff.min()
            min_pos = np.argmin(diff)
            recall_counter += (min_dist <= tolerance)

        pred_counter += len(yhat)
        gt_counter += len(y)
        
    p, r, f1, rval = get_metrics(precision_counter,
                                      recall_counter,
                                      pred_counter,
                                      gt_counter)
    return p, r, f1, rval
    
def create_phone_mask(lengths):

    mask = torch.zeros(len(lengths),sum(lengths))

    cumsum = 0
    for i,l in enumerate(lengths):
        mask[i,cumsum:cumsum+l] = 1/l+1e-8
        cumsum += l

    return mask

def create_sliding_window(lengths,win_len=10,shift=5):

    L = torch.max(lengths)
    num_win = torch.ceil(L/shift).long()
    sliding_window = torch.zeros(num_win,L)
    for n in range(num_win):
        sliding_window[n,n*shift:min(n*shift+win_len,L)] = 1.0

    return sliding_window

def forced_align(cost,phone_feature):
    
    D,align = dtw(C=cost.T,
                  step_sizes_sigma=np.array([[1, 1], [0,1]]))
    #print(align)
    align_seq = [-1 for i in range(max(align[:,0])+1)]
    for i in list(align):
        
        if align_seq[i[0]]<i[1]:
            align_seq[i[0]]=i[1]
    align_id = list(align_seq)
    
    return [(i,p) for i,p in zip(align_id[:-1],phone_feature[1:-1])]

    
def TIMIT_eval(dataset,speech_encoder, phone_encoder, tokenizer, processor,unit='phone',win_len=1,frame_shift=1,tolerance=0.02,embed_only=False):
    
    device = speech_encoder.device
    
    evaluation_pairs = []
    
    for entry in tqdm(dataset):
        
        if unit == 'word':
            y = entry['word_detail']['start']
            y_ids = entry['word_detail']['utterance']
        elif unit == 'phone':
            y = entry['merge_phonetic_detail']['start']
            y_ids = entry['merge_phonetic_detail']['utterance']            
        
        
        input_features = [entry['audio']['array']]

        batch = processor(input_features, sampling_rate=16000, return_attention_mask=True,return_tensors="pt")
        
        length = torch.max(torch.sum(batch['attention_mask'],dim=-1))
        batch['attention_mask'] = batch['attention_mask'][:,:length]
        batch['input_features'] = batch['input_features'][:,:,:length]
        speech_lengths = torch.sum(batch['attention_mask'][:,::2],dim=-1)
        speech_sliding_mask = create_sliding_window(speech_lengths,win_len=win_len,shift=frame_shift)
        
        if unit == 'word':
            augmented_y_ids = ['[SEP]'] + y_ids + ['[SEP]']
        elif unit == 'phone':
            augmented_y_ids = y_ids
            y = y[1:-1]
            
        out = tokenizer(augmented_y_ids, return_attention_mask=False, return_length=True, return_token_type_ids=False, add_special_tokens=False)
        phones = torch.tensor(list(itertools.chain.from_iterable(out['input_ids']))).long().unsqueeze(0)
        phone_mask = create_phone_mask(out['length'])
        
        with torch.no_grad():
            speech_features = speech_encoder(**batch.to(device)).last_hidden_state.squeeze()
            if embed_only:
                phone_features = phone_encoder.bert.embeddings(phones.to(device)).squeeze()
            else:
                phone_features = phone_encoder(phones.to(device)).last_hidden_state.squeeze()

        transformed_phone_features = torch.matmul(phone_mask.to(device),phone_features)
        transformed_speech_features = torch.matmul(speech_sliding_mask.to(device),speech_features)
        transformed_speech_features = F.normalize(transformed_speech_features,dim=-1)
        transformed_phone_features = F.normalize(transformed_phone_features,dim=-1)
        pairwise_cos_sim = torch.matmul(transformed_speech_features,transformed_phone_features.t())
        

        alignment = forced_align(-pairwise_cos_sim.cpu().numpy(),augmented_y_ids)
        #print(alignment)
        #alignment = list(torch.argmax(pairwise_cos_sim,dim=1).cpu().numpy())
        y_hat = [i*frame_shift*0.02 for i,p in alignment]
        y_hat_ids = [p for i,p in alignment]
        
        print(y_hat)
        print(y_hat_ids)
        print(y)
        print(augmented_y_ids[1:-1])
        evaluation_pairs.append(((np.array(y),np.array(augmented_y_ids[1:-1])), (np.array(y_hat), np.array(y_hat_ids))))
        
    p, r, f1, rval = get_all_stats(evaluation_pairs,tolerance=tolerance)
    print(p,r,f1,rval)
    return p,r,f1,rval


def DoReCo_eval(dataset, speech_encoder, phone_encoder, tokenizer, processor,unit='phone',win_len=1,frame_shift=1,tolerance=0.02,embed_only=False):
    
    device = speech_encoder.device
    
    evaluation_pairs = []
    
    for entry in tqdm(dataset):
        
        try:
        
            if unit == 'word':
                y = entry['word_details']['start']
                y_ids = entry['word_details']['word']
            elif unit == 'phone':
                y = entry['phone_details']['start']
                y_ids = entry['phone_details']['word']            


            input_features = [entry['audio']['array']]

            batch = processor(input_features, sampling_rate=16000, return_attention_mask=True,return_tensors="pt")

            length = torch.max(torch.sum(batch['attention_mask'],dim=-1))
            batch['attention_mask'] = batch['attention_mask'][:,:length]
            batch['input_features'] = batch['input_features'][:,:,:length]
            speech_lengths = torch.sum(batch['attention_mask'][:,::2],dim=-1)
            speech_sliding_mask = create_sliding_window(speech_lengths,win_len=win_len,shift=frame_shift)

            augmented_y_ids = [i if i!='[SIL]' else '[SEP]' for i in y_ids]

            out = tokenizer(augmented_y_ids, return_attention_mask=False, return_length=True, return_token_type_ids=False, add_special_tokens=False)
            phones = torch.tensor(list(itertools.chain.from_iterable(out['input_ids']))).long().unsqueeze(0)
            phone_mask = create_phone_mask(out['length'])

            with torch.no_grad():
                speech_features = speech_encoder(**batch.to(device)).last_hidden_state.squeeze()
                if embed_only:
                    phone_features = phone_encoder.bert.embeddings(phones.to(device)).squeeze()
                else:
                    phone_features = phone_encoder(phones.to(device)).last_hidden_state.squeeze()

            transformed_phone_features = torch.matmul(phone_mask.to(device),phone_features)
            transformed_speech_features = torch.matmul(speech_sliding_mask.to(device),speech_features)
            transformed_speech_features = F.normalize(transformed_speech_features,dim=-1)
            transformed_phone_features = F.normalize(transformed_phone_features,dim=-1)
            pairwise_cos_sim = torch.matmul(transformed_speech_features,transformed_phone_features.t())


            alignment = forced_align(-pairwise_cos_sim.cpu().numpy(),augmented_y_ids)
            #print(alignment)
            #alignment = list(torch.argmax(pairwise_cos_sim,dim=1).cpu().numpy())
            y_hat = [i*frame_shift*0.02 for i,p in alignment]
            y_hat_ids = [p for i,p in alignment]

            print(y_hat)
            print(y_hat_ids)
            print(y[1:-1])
            print(augmented_y_ids[1:-1])
            evaluation_pairs.append(((np.array(y[1:-1]),np.array(augmented_y_ids[1:-1])), (np.array(y_hat), np.array(y_hat_ids))))
        except:
            continue
        
    p, r, f1, rval = get_all_stats(evaluation_pairs,tolerance=tolerance)
    print(p,r,f1,rval)
    return p,r,f1,rval


def load_models(checkpoint_path):

    checkpoints = torch.load(checkpoint_path)
    checkpoints['hyper_parameters']['speech_encoder_cfg']['pooling'] = 'mean'
    
    speech_encoder = clap.encoders.initialize_speech_model(checkpoints['hyper_parameters']['speech_encoder_cfg'])
    phone_encoder = clap.encoders.initialize_phone_model(checkpoints['hyper_parameters']['phone_encoder_cfg'])

    speech_encoder.load_state_dict({k.replace('speech_encoder.',''):v for k,v in checkpoints['state_dict'].items() if 'speech_encoder' in k},strict=False)
    phone_encoder.load_state_dict({k.replace('phone_encoder.',''):v for k,v in checkpoints['state_dict'].items() if 'phone_encoder' in k},strict=False)

    speech_encoder.config.pooling = checkpoints['hyper_parameters']['speech_encoder_cfg']['pooling']
    print(checkpoints['hyper_parameters']['speech_encoder_cfg']['pooling'])
    
    return phone_encoder.to('cuda').eval(), speech_encoder.to('cuda').eval()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=str)
    parser.add_argument('--speech_encoder',type=str)
    parser.add_argument('--phone_encoder',type=str)
    parser.add_argument('--name',type=str)
    parser.add_argument('--cache_dir',type=str,default='/home/lingjzhu/scratch/cache')
    args = parser.parse_args()
    
    run = wandb.init(
                # Set the project where this run will be logged
                project="clap-align-eval",
                name=args.name
                )
    device = 'cuda'
    
    processor = AutoProcessor.from_pretrained('openai/whisper-base')
    tokenizer = DebertaV2Tokenizer.from_pretrained('charsiu/IPATokenizer')
    
    
    if args.checkpoint is not None:
        phone_encoder, speech_encoder = load_models(args.checkpoint)
    else:
        phone_encoder = PhoneEncoder.from_pretrained(args.phone_encoder)
        speech_encoder = SpeechEncoder.from_pretrained(args.speech_encoder)
        
    phone_encoder.eval().to(device)
    speech_encoder.eval().to(device)
    
    dataset = load_dataset('anyspeech/frame_labels',cache_dir=args.cache_dir)['test']
    
    p,r,f1,rval = TIMIT_eval(dataset,speech_encoder, phone_encoder, tokenizer, processor,unit='phone',win_len=1,frame_shift=1,tolerance=0.02,embed_only=False)
    
    wandb.log({'TIMIT-phone-Precision':p,
               'TIMIT-phone-Recall':r,
               'TIMIT-phone-F1':f1,
               'TIMIT-phone-R-value':rval})
    
    p,r,f1,rval = TIMIT_eval(dataset,speech_encoder, phone_encoder, tokenizer, processor,unit='word',win_len=3,frame_shift=2,tolerance=0.1,embed_only=False)
    
    wandb.log({'TIMIT-word-Precision':p,
               'TIMIT-word-Recall':r,
               'TIMIT-word-F1':f1,
               'TIMIT-word-R-value':rval})
    
    doreco = load_dataset('anyspeech/fw_frame_labels',cache_dir=args.cache_dir)
    
    langs = list(doreco.keys())
    
    
    seen_aggregate = defaultdict(list)
    unseen_aggregate = defaultdict(list)
    
    unseen = set(['ana1239', 'beja1238', 'cabe1245', 'dolg1241', 'stan1290', 'jeha1242', 'light_warlpiri_ligh1234.tar', 'sout2856', 'nisv1234', 'nort2641', 'pnar1238', 'urum1249', 'warl1254', 'apah1238'])
    print(langs)
    
    for lang in langs:
        dataset = doreco[lang]
        p,r,f1,rval = DoReCo_eval(dataset,speech_encoder, phone_encoder, tokenizer, processor,unit='word',win_len=3,frame_shift=2,tolerance=0.1)
        wandb.log({'%s/word-Precision'%lang:p,
           '%s/word-Recall'%lang:r,
           '%s/word-F1'%lang:f1,
           '%s/word-R-value'%lang:rval})
        
        if lang in unseen:
            unseen_aggregate['word-Precision'].append(p)
            unseen_aggregate['word-Recall'].append(r)
            unseen_aggregate['word-F1'].append(f1)
            unseen_aggregate['word-R-value'].append(rval)
        else:
            seen_aggregate['word-Precision'].append(p)
            seen_aggregate['word-Recall'].append(r)
            seen_aggregate['word-F1'].append(f1)
            seen_aggregate['word-R-value'].append(rval)
            
        
        p,r,f1,rval = DoReCo_eval(dataset,speech_encoder, phone_encoder, tokenizer, processor,unit='phone',win_len=1,frame_shift=1,tolerance=0.02)
        wandb.log({'%s/phone-Precision'%lang:p,
           '%s/phone-Recall'%lang:r,
           '%s/phone-F1'%lang:f1,
           '%s/phone-R-value'%lang:rval})
        
        if lang in unseen:
            unseen_aggregate['phone-Precision'].append(p)
            unseen_aggregate['phone-Recall'].append(r)
            unseen_aggregate['phone-F1'].append(f1)
            unseen_aggregate['phone-R-value'].append(rval)
        else:
            seen_aggregate['phone-Precision'].append(p)
            seen_aggregate['phone-Recall'].append(r)
            seen_aggregate['phone-F1'].append(f1)
            seen_aggregate['phone-R-value'].append(rval)
            
    wandb.log({'Doreco-seen/word-Precision':np.mean(seen_aggregate['word-Precision']),
               'Doreco-seen/word-Recall':np.mean(seen_aggregate['word-Recall']),
               'Doreco-seen/word-F1':np.mean(seen_aggregate['word-F1']),
               'Doreco-seen/word-R-value':np.mean(seen_aggregate['word-R-value']),
               'Doreco-unseen/word-Precision':np.mean(unseen_aggregate['word-Precision']),
               'Doreco-unseen/word-Recall':np.mean(unseen_aggregate['word-Recall']),
               'Doreco-unseen/word-F1':np.mean(unseen_aggregate['word-F1']),
               'Doreco-unseen/word-R-value':np.mean(unseen_aggregate['word-R-value']),
               'Doreco-seen/phone-Precision':np.mean(seen_aggregate['phone-Precision']),
               'Doreco-seen/phone-Recall':np.mean(seen_aggregate['phone-Recall']),
               'Doreco-seen/phone-F1':np.mean(seen_aggregate['phone-F1']),
               'Doreco-seen/phone-R-value':np.mean(seen_aggregate['phone-R-value']),
               'Doreco-unseen/phone-Precision':np.mean(unseen_aggregate['phone-Precision']),
               'Doreco-unseen/phone-Recall':np.mean(unseen_aggregate['phone-Recall']),
               'Doreco-unseen/phone-F1':np.mean(unseen_aggregate['phone-F1']),
               'Doreco-unseen/phone-R-value':np.mean(unseen_aggregate['phone-R-value']),              
              })