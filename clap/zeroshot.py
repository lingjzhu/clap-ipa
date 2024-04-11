from datasets import load_dataset
import torch
import re
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import clap.encoders

def Google_Command_Eval(speech_encoder, phone_encoder, tokenizer, processor, text=False):
    
    G = load_dataset("speech_commands", "v0.01", split='test',cache_dir='/home/lingjzhu/scratch/cache')
    
    speech = [entry['audio']['array'] for entry in G]
    labels = [re.search(r'(.*)/.*',entry['file']).group(1) for entry in G] 
    
    label_mapping = {'_silence_': 'silence','bed': 'ˈbɛd', 'bird': 'ˈbɝd', 'cat': 'ˈkæt', 'dog': 'ˈdɔɡ', 'down': 'ˈdaʊn',
     'eight': 'ˈeɪt', 'five': 'ˈfaɪv', 'four': 'ˈfɔɹ', 'go': 'ˈɡoʊ', 'happy': 'ˈhæpi', 'house': 'ˈhaʊs', 'left': 'ˈɫɛft', 'marvin': 'ˈmɑɹvɪn',
    'nine': 'ˈnaɪn', 'no': 'ˈnoʊ', 'off': 'ˈɔf', 'on': 'ˈɔn', 'one': 'ˈwən', 'right': 'ˈɹaɪt', 'seven': 'ˈsɛvən', 'sheila': 'ˈʃiɫə',
     'six': 'ˈsɪks', 'stop': 'ˈstɑp', 'three': 'ˈθɹi', 'tree': 'ˈtɹi', 'two': 'ˈtu', 'up': 'ˈəp', 'wow': 'ˈwaʊ', 'yes': 'ˈjɛs','zero': 'ˈzɪɹoʊ'}
    
    labels = [label_mapping[l] for l in labels]
    gt = [i for i in set(labels) if i != 'silence']

    dtype = speech_encoder.dtype
    pembs = []
    for g in gt:
        tokens = tokenizer(g,return_token_type_ids=False, return_tensors='pt')
        with torch.no_grad():
            out = phone_encoder(**tokens.to('cuda'))
            emb = out.pooler_output
        pembs.append(emb.cpu().squeeze())

    pembs = torch.stack(pembs).to(torch.float32)

    sembs = []
    for s in tqdm(speech):
        audio = processor(s,return_attention_mask=True,sampling_rate=16000,max_length=32000,return_tensors='pt')
        #audio['attention_mask'] = audio['attention_mask']
        audio['input_features'] = audio['input_features'].to(dtype)
        with torch.no_grad():
            emb = speech_encoder(**audio.to('cuda')).pooler_output
        sembs.append(emb.cpu().squeeze())

    sembs = torch.stack(sembs).to(torch.float32)

    sembs = F.normalize(sembs,dim=-1)
    pembs = F.normalize(pembs,dim=-1)

    outs = []
    for s,l in tqdm(zip(sembs,labels)):

        scores = torch.matmul(pembs,s)
        sorted_indices = np.argsort(-scores)
        sorted_target_labels = np.array(gt)[sorted_indices]

        if scores[sorted_indices[0]] < 0.10: # this change constantly across models
            if 'silence' == l:
                pred_rank = 0
            else:
                pred_rank = 100
        else:
            try:
                pred_rank = np.where(sorted_target_labels == l)[0].item()
            except:
                pred_rank = 100

        if pred_rank == 0:
            outs.append(1)
        else:
            outs.append(0)

    return np.mean(outs)

