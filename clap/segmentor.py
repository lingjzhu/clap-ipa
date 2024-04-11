import math

import numpy as np
import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import transformers
from transformers import AutoModel,DebertaV2Tokenizer, AutoProcessor
from torchmetrics.functional import pairwise_cosine_similarity
from tslearn.metrics import SoftDTWLossPyTorch
import wandb
import matplotlib.pyplot as plt

import clap.encoders
import clap.losses
from clap.boundary_eval import TIMIT_eval

class ConvBank(nn.Module):
    def __init__(self, input_dim, output_class_num, kernels, cnn_size, hidden_size, dropout, **kwargs):
        super(ConvBank, self).__init__()
        self.drop_p = dropout
        
        self.in_linear = nn.Linear(input_dim, hidden_size)
        latest_size = hidden_size

        # conv bank
        self.cnns = nn.ModuleList()
        assert len(kernels) > 0
        for kernel in kernels:
            self.cnns.append(nn.Conv1d(latest_size, cnn_size, kernel, padding=kernel//2))
        latest_size = cnn_size * len(kernels)

        self.out_linear = nn.Linear(latest_size, output_class_num)

    def forward(self, features):
        hidden = F.dropout(F.relu(self.in_linear(features)), p=self.drop_p)

        conv_feats = []
        hidden = hidden.transpose(1, 2).contiguous()
        for cnn in self.cnns:   
            conv_feats.append(cnn(hidden))
        hidden = torch.cat(conv_feats, dim=1).transpose(1, 2).contiguous()
        hidden = F.dropout(F.relu(hidden), p=self.drop_p)

        predicted = self.out_linear(hidden)
        return predicted
    

class SegmentExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        speech_encoder_cfg,
        phone_encoder_cfg,
        optimizer,
        sample_rate: int = 16000,
        initial_lr: float = 1e-4,
        weight_decay: float = 1e-4,
        num_warmup_steps: int = 0,
        gamma: float = 0.1,
        temperature: float = 0.05,
        tokenizer = None,
        processor = None,
        zeroshot = None, 
        freeze_phone_encoder = None,
        freeze_speech_encoder = None,
        use_ctc_loss = None,
        use_l1_loss = None,
        use_contrastive_loss = None,
        loss = 'fs_blank'
    ):

        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)

        self.speech_encoder = clap.encoders.initialize_speech_model(self.hparams.speech_encoder_cfg)
        self.phone_encoder = clap.encoders.initialize_phone_model(self.hparams.phone_encoder_cfg)
        
        if freeze_speech_encoder:
            for param in self.speech_encoder.encoder.parameters():
                param.requires_grad = False
                
        if freeze_phone_encoder:
            for param in self.phone_encoder.bert.parameters():
                param.requires_grad = False
        
        if loss == 'fs':
            self.loss_fn = clap.losses.ForwardSumLoss()
        elif loss == 'fs_blank':
            self.loss_fn = clap.losses.ForwardSumLossWithBlank()
        
        self.validation_step_outputs = []
        
        
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer)
        
        self.temperature = temperature
        
        if processor is not None:
            self.processor = AutoProcessor.from_pretrained(processor)
            
        self.use_contrastive_loss = use_contrastive_loss
        if self.use_contrastive_loss:
            self.siglip_loss = clap.losses.ContrastiveSigmoid
        
        self.use_ctc_loss = use_ctc_loss
        if self.use_ctc_loss:
            self.proj_size = self.hparams.speech_encoder_cfg['projection_size']
            self.vocab_size = self.hparams.phone_encoder_cfg['vocab_size']
            self.ctc_decoder = nn.Sequential(nn.Linear(self.proj_size,self.vocab_size))
            
        self.use_l1_loss = use_l1_loss
        if self.use_l1_loss:
            self.proj_size = self.hparams.speech_encoder_cfg['projection_size']
            if self.use_ctc_loss:
                self.input_size = self.vocab_size
            else:
                self.input_size = self.proj_size
            self.mel_decoder = ConvBank(self.input_size,80,[5,3],self.proj_size,self.proj_size,0.1)
            self.l1_loss = nn.L1Loss(reduction='sum')
        

    def configure_optimizers(self):
        model_params = [
            {"params": self.speech_encoder.parameters()},
            {"params": self.phone_encoder.parameters()},
        ]
        
        optimizer = torch.optim.AdamW(model_params, lr=self.hparams.initial_lr,weight_decay=self.hparams.weight_decay)

        max_steps = self.trainer.max_steps  # Max steps per optimizer

        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )
        
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step"}],
        )

    def forward(self, audio_input, phone_input):
        speech_features, speech_emb = self.encode_speech(audio_input)
        phone_features, phone_emb = self.encode_phones(phone_input)
        return speech_features, phone_features, speech_emb, phone_emb
    
    def encode_speech(self,audio_input):
        speech_features = self.speech_encoder(**audio_input)
        return speech_features.last_hidden_state, speech_features.pooler_output
    
    def encode_phones(self, phone_input):
        phone_features = self.phone_encoder(**phone_input)
        return phone_features.last_hidden_state, phone_features.pooler_output
    
    
    def training_step(self, batch, batch_idx, **kwargs):
        
        audio_input, phone_input, speech_lengths, phone_lengths, phone_masks, ctc_targets = batch

        speech_features, phone_features, speech_emb, phone_emb = self(audio_input, phone_input)
        #print(phone_masks)
        # batch x 1 x max(mel_lens) x max(text_lens)
        transformed_phone_features = torch.bmm(phone_masks,phone_features)
        #transformed_speech_features = torch.bmm(speech_sliding_mask,speech_features)
        transformed_speech_features = F.normalize(speech_features,dim=-1)
        transformed_phone_features = F.normalize(transformed_phone_features,dim=-1)
        pairwise_cos_sim = torch.bmm(transformed_speech_features,transformed_phone_features.transpose(1,2)).unsqueeze(1)
        pairwise_cos_sim = pairwise_cos_sim/self.temperature
        loss = self.loss_fn(pairwise_cos_sim,phone_lengths,speech_lengths)

        self.log("forwardsum_loss", loss, prog_bar=True)

        if self.use_contrastive_loss:
            closs = self.siglip_loss(speech_emb,
                             phone_emb,
                             self.phone_encoder.t_prime,
                             self.phone_encoder.b)
            
            self.log("contrastive_loss", closs, prog_bar=True)
            loss = 0.1*loss + closs
            
        if self.use_ctc_loss:
            logits = self.ctc_decoder(speech_features)

            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            input_lengths = torch.sum(audio_input['attention_mask'][:,::2],dim=-1)

            with torch.backends.cudnn.flags(enabled=False):
                ctc_loss = nn.functional.ctc_loss(
                    log_probs,
                    ctc_targets['input_ids'],
                    input_lengths,
                    ctc_targets['length'],
                    blank=self.tokenizer.eos_token_id,
                    reduction='mean',
                    zero_infinity=True,
                )

            self.log("ctc_loss", ctc_loss, prog_bar=True)
            loss += 0.1*ctc_loss
            
        if self.use_l1_loss:
            if self.use_ctc_loss:
                pred_mels = self.mel_decoder(log_probs.transpose(0, 1))
            else:
                pred_mels = self.mel_decoder(speech_features)
            
            l1_loss = self.l1_loss(pred_mels*audio_input['attention_mask'][:,::2].unsqueeze(-1), audio_input['input_features'][:,:,::2].transpose(1,2)*audio_input['attention_mask'][:,::2].unsqueeze(-1))
            l1_loss = l1_loss/torch.sum(audio_input['attention_mask'][:,::2])
            self.log("l1_loss", l1_loss, prog_bar=True)
            loss += 0.1*l1_loss
                   
            
        return loss


    def validation_step(self, batch, batch_idx, **kwargs):
        #print(batch)
        audio_input, phone_input, speech_lengths, phone_lengths, phone_masks,ctc_targets = batch
        
        speech_features, phone_features, speech_emb, phone_emb = self(audio_input, phone_input)
        #print(phone_masks)
        # batch x 1 x max(mel_lens) x max(text_lens)
        transformed_phone_features = torch.bmm(phone_masks,phone_features)
        #transformed_speech_features = torch.bmm(speech_sliding_mask,speech_features)
        transformed_speech_features = F.normalize(speech_features,dim=-1)
        transformed_phone_features = F.normalize(transformed_phone_features,dim=-1)
        pairwise_cos_sim = torch.bmm(transformed_speech_features,transformed_phone_features.transpose(1,2)).unsqueeze(1)
        pairwise_cos_sim = pairwise_cos_sim/self.temperature
        loss = self.loss_fn(pairwise_cos_sim,phone_lengths,speech_lengths)

        self.validation_step_outputs.append({'val_loss':loss, 'alignment':pairwise_cos_sim.squeeze().cpu().numpy(), 'phone_len':phone_lengths, 'speech_len':speech_lengths})
        
            
        

    def on_validation_epoch_end(self):
        
        if self.global_rank == 0:

            avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
            
            self.log("val_loss", avg_loss, sync_dist=True)

            fig1 = plt.figure()
            data = self.validation_step_outputs[0]['alignment'][0]
            data = data[:self.validation_step_outputs[0]['speech_len'][0],:self.validation_step_outputs[0]['phone_len'][0]]
            plt.imshow(data.T, interpolation='nearest')
            plt.tight_layout()
            self.logger.experiment.log({"Alignment-1":fig1})

            fig2 = plt.figure()
            data = self.validation_step_outputs[1]['alignment'][0]
            data = data[:self.validation_step_outputs[1]['speech_len'][0],:self.validation_step_outputs[1]['phone_len'][0]]
            plt.imshow(data.T, interpolation='nearest')
            plt.tight_layout()
            self.logger.experiment.log({"Alignment-2":fig2})
        
        
            fig3 = plt.figure()
            data = self.validation_step_outputs[1]['alignment'][3]
            data = data[:self.validation_step_outputs[1]['speech_len'][3],:self.validation_step_outputs[1]['phone_len'][3]]
            plt.imshow(data.T, interpolation='nearest')
            plt.tight_layout()
            self.logger.experiment.log({"Alignment-3":fig3})
            
            
        
            fig4 = plt.figure()
            data = self.validation_step_outputs[0]['alignment'][3]
            data = data[:self.validation_step_outputs[0]['speech_len'][3],:self.validation_step_outputs[0]['phone_len'][3]]
            plt.imshow(data.T, interpolation='nearest')
            plt.tight_layout()
            self.logger.experiment.log({"Alignment-4":fig4})

            self.validation_step_outputs.clear()
            
            p,r,f1,rval = TIMIT_eval(self.speech_encoder, self.phone_encoder, self.tokenizer, self.processor)
            
            self.log('precision',p)
            self.log('recall',r)
            self.log('f1',f1)
            self.log('rval',rval)
            
            
            
            
            
                



            