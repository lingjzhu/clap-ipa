import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from datasets import load_dataset
from transformers import DebertaV2Tokenizer, AutoProcessor, AutoTokenizer
import argparse
import wandb
from torchmetrics.functional.retrieval import *

import clap.encoders

def compute_ranking_metrics(
    queries, 
    targets,
    query_labels,
    target_labels,
    metric='cosine',
    ):
    num_queries = len(query_labels)

    #print(queries.shape)
    #print(targets.shape)
    distances = pairwise_distances(queries, Y=targets, metric=metric, n_jobs=-1)
    distances = -(distances-1)
    
    mAP = []
    hit = []
    hit10 = []

    for i in range(num_queries):

        dist = torch.tensor(distances[i])

        y = torch.tensor(np.where(query_labels[i] == target_labels, 1.0, 0.0))

        
        mAP.append(retrieval_average_precision(dist, y))
        hit.append(retrieval_hit_rate(dist,y,top_k=1))
        hit10.append(retrieval_hit_rate(dist,y,top_k=10))
        
    return_dict = {
        'mAP': torch.mean(torch.tensor(mAP)),
        'hit@1': torch.mean(torch.tensor(hit)),
        'hit@10': torch.mean(torch.tensor(hit10))
    }

    return return_dict


def p2s_evaluate(speech_encoder, phone_encoder, tokenizer, processor, data, args):

    if args.data=='fleurs' and args.text:
        queries = set([i['transcription'] for i in data['query']])
        query_labels = queries
        candidates = [i['audio']['array'] for i in data['candidate']]
        candidates_labels = [i['transcription'] for i in data['candidate']]
        
    elif args.data=='mswc' and args.text:
        queries = set([i['key'].split('_')[2] for i in data['query']])
        query_labels = queries
        candidates = [i['audio']['array'] for i in data['candidate']]
        candidates_labels = [i['key'].split('_')[2] for i in data['candidate']]
    
    else:
        queries = set([i['phones'] for i in data['query']])
        query_labels = queries
        candidates = [i['audio']['array'] for i in data['candidate']]
        candidates_labels = [i['phones'] for i in data['candidate']]
        
        
    print('Evaluating...')
    all_queries = []
    all_targets = []
    all_query_labels = []
    all_target_labels = []
    
 
    for q,ql in tqdm(zip(queries, query_labels)):
        with torch.no_grad():
            batch = tokenizer(q,max_length=512,truncation=True,padding=True,                                           return_token_type_ids=False,return_tensors="pt")
            query = phone_encoder(**batch.to(device)).pooler_output
            all_queries.append(query.cpu().detach().numpy())
            all_query_labels.append(ql)

    for t,tl in tqdm(zip(candidates, candidates_labels)):
        with torch.no_grad():
            batch = processor(t,max_length=480000, sampling_rate=16000, return_attention_mask=True,return_tensors="pt")
            target = speech_encoder(**batch.to(device)).pooler_output
            all_targets.append(target.cpu().detach().numpy())
            all_target_labels.append(tl)


    all_target_labels = np.array(all_target_labels)
    all_query_labels = np.array(all_query_labels)
    all_queries = np.concatenate(all_queries,axis=0)
    all_targets = np.concatenate(all_targets,axis=0)
    
    return compute_ranking_metrics(all_queries,all_targets,all_query_labels,all_target_labels)



def s2p_evaluate(speech_encoder, phone_encoder, tokenizer, processor, data, args):

    
    if args.data=='fleurs' and args.text:
        queries = [i['audio']['array'] for i in data['query']]
        query_labels = [i['transcription'] for i in data['query']]
        candidates = set([i['transcription'] for i in data['candidate']])
        candidates_labels = candidates
    elif args.data=='mswc' and args.text:
        queries = [i['audio']['array'] for i in data['query']]
        query_labels = [i['key'].split('_')[2] for i in data['query']]
        candidates = set([i['key'].split('_')[2] for i in data['candidate']])
        candidates_labels = candidates        
    else:
        queries = [i['audio']['array'] for i in data['query']]
        query_labels = [i['phones'] for i in data['query']]
        candidates = set([i['phones'] for i in data['candidate']])
        candidates_labels = candidates

    print('Evaluating...')
    all_queries = []
    all_targets = []
    all_query_labels = []
    all_target_labels = []
    
 
    for q,ql in tqdm(zip(queries, query_labels)):
        with torch.no_grad():
            batch = processor(q,max_length=480000, sampling_rate=16000, return_attention_mask=True,return_tensors="pt")
            query = speech_encoder(**batch.to(device)).pooler_output
            all_queries.append(query.cpu().detach().numpy())
            all_query_labels.append(ql)

    for t,tl in tqdm(zip(candidates, candidates_labels)):
        with torch.no_grad():
            batch = tokenizer(t,max_length=512,truncation=True,padding=True,                                           return_token_type_ids=False,return_tensors="pt")
            target = phone_encoder(**batch.to(device)).pooler_output
            all_targets.append(target.cpu().detach().numpy())
            all_target_labels.append(tl)


    all_target_labels = np.array(all_target_labels)
    all_query_labels = np.array(all_query_labels)
    all_queries = np.concatenate(all_queries,axis=0)
    all_targets = np.concatenate(all_targets,axis=0)
    
    return compute_ranking_metrics(all_queries,all_targets,all_query_labels,all_target_labels)


def s2s_evaluate(speech_encoder, phone_encoder, tokenizer, processor, data, args):

    queries = [i['audio']['array'] for i in data['query']]

    query_labels = [i['phones'] for i in data['query']]

    candidates = [i['audio']['array'] for i in data['candidate']]

    candidates_labels = [i['phones'] for i in data['candidate']]


    print('Evaluating...')
    all_queries = []
    all_targets = []
    all_query_labels = []
    all_target_labels = []
    
 
    for q,ql in tqdm(zip(queries, query_labels)):
        with torch.no_grad():
            batch = processor(q,max_length=480000, sampling_rate=16000, return_attention_mask=True,return_tensors="pt")
            query = speech_encoder(**batch.to(device)).pooler_output
            all_queries.append(query.cpu().detach().numpy())
            all_query_labels.append(ql)

    for t,tl in tqdm(zip(candidates, candidates_labels)):
        with torch.no_grad():
            batch = processor(t,max_length=480000, sampling_rate=16000, return_attention_mask=True,return_tensors="pt")
            target = speech_encoder(**batch.to(device)).pooler_output
            all_targets.append(target.cpu().detach().numpy())
            all_target_labels.append(tl)

    all_target_labels = np.array(all_target_labels)
    all_query_labels = np.array(all_query_labels)
    all_queries = np.concatenate(all_queries,axis=0)
    all_targets = np.concatenate(all_targets,axis=0)
    
    return compute_ranking_metrics(all_queries,all_targets,all_query_labels,all_target_labels)



def load_models(checkpoint_path):

    checkpoints = torch.load(checkpoint_path)
    
    speech_encoder = clap.encoders.initialize_speech_model(checkpoints['hyper_parameters']['speech_encoder_cfg'])
    phone_encoder = clap.encoders.initialize_phone_model(checkpoints['hyper_parameters']['phone_encoder_cfg'])

    speech_encoder.load_state_dict({k.replace('speech_encoder.',''):v for k,v in checkpoints['state_dict'].items() if 'speech_encoder' in k},strict=True)
    phone_encoder.load_state_dict({k.replace('phone_encoder.',''):v for k,v in checkpoints['state_dict'].items() if 'phone_encoder' in k},strict=True)

    return phone_encoder.to('cuda').eval(), speech_encoder.to('cuda').eval()


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer',type=str,default='charsiu/IPATokenizer')
    parser.add_argument('--data',type=str, default='mswc')
    parser.add_argument('--checkpoint',type=str)
    parser.add_argument('--name',type=str)
    parser.add_argument('--pooling',type=str,default='mean')
    parser.add_argument('--text',action='store_true')
    args = parser.parse_args()
    
    
    run = wandb.init(
                # Set the project where this run will be logged
                project="clap-ipa-eval",
                name=args.name
                )

    device = 'cuda'
    
    phone_encoder, speech_encoder = load_models(args.checkpoint)
    
    
    processor = AutoProcessor.from_pretrained('openai/whisper-base')
    if args.text:
        print('Using text tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.tokenizer)
    
    if args.data == 'mswc':
        
        data = load_dataset('anyspeech/mswc_test',cache_dir='/home/lingjzhu/scratch/cache')
        
    elif args.data == 'fleurs':
        
        data = load_dataset('anyspeech/fleurs_test',cache_dir='/home/lingjzhu/scratch/cache')

    s2s_score = s2s_evaluate(speech_encoder, phone_encoder, tokenizer, processor, data, args)
    
    s2s_score = {args.data+'_'+'s2s/'+k:v for k,v in s2s_score.items()}
    wandb.log(s2s_score)
    
    s2p_score = s2p_evaluate(speech_encoder, phone_encoder, tokenizer, processor, data, args)
    
    s2p_score = {args.data+'_'+'s2p/'+k:v for k,v in s2p_score.items()}
    wandb.log(s2p_score)
    
    p2s_score = p2s_evaluate(speech_encoder, phone_encoder, tokenizer, processor, data, args)
    
    p2s_score = {args.data+'_'+'p2s/'+k:v for k,v in p2s_score.items()}
    wandb.log(p2s_score)
    
    
    
    #data = load_dataset('anyspeech/fleurs_test',cache_dir='/scratch/lingjzhu_root/lingjzhu1/lingjzhu/cache')
    
   