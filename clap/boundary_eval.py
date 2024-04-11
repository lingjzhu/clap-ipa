#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm 
from datasets import load_dataset
from lingpy.sequence.sound_classes import ipa2tokens
from librosa.sequence import dtw
import torch
import torch.nn.functional as F

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

    
def TIMIT_eval(speech_encoder, phone_encoder, tokenizer, processor,unit='phone',win_len=1,frame_shift=1,tolerance=0.02):
    
    device = speech_encoder.device
    dataset = load_dataset('anyspeech/frame_labels')['train'] # use the training set for valiation
    
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
            
        out = tokenizer(augmented_y_ids, return_attention_mask=False, return_length=True, return_token_type_ids=False, add_special_tokens=False)
        phones = torch.tensor(list(itertools.chain.from_iterable(out['input_ids']))).long().unsqueeze(0)
        phone_mask = create_phone_mask(out['length'])
        
        with torch.no_grad():
            speech_features = speech_encoder(**batch.to(device)).last_hidden_state.squeeze()
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
        
    p, r, f1, rval = get_all_stats(evaluation_pairs,tolerance=tolerance)
    print(p,r,f1,rval)
    return p,r,f1,rval