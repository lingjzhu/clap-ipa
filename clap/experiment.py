import math

import numpy as np
import lightning as pl
import torch
import torchaudio
import transformers
from transformers import AutoModel,DebertaV2Tokenizer, AutoProcessor, AutoTokenizer

import clap.encoders
import clap.losses
from clap.zeroshot import Google_Command_Eval,UCLA_Phonetic_Corpus_Eval

class CLAPExp(pl.LightningModule):
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
        hard_negatives: bool = False,
        tokenizer = None,
        processor = None,
        zeroshot = None,
        text = False
    ):

        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)

        self.speech_encoder = clap.encoders.initialize_speech_model(self.hparams.speech_encoder_cfg)
        self.phone_encoder = clap.encoders.initialize_phone_model(self.hparams.phone_encoder_cfg)

        self.loss = clap.losses.ContrastiveSigmoid
        self.hard_negatives = hard_negatives
        self.text = text
        self.validation_step_outputs = []
        
        if tokenizer is not None:
            if tokenizer == 'xlm-roberta-base':
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            else:
                self.tokenizer = DebertaV2Tokenizer.from_pretrained(tokenizer)
        if processor is not None:
            self.processor = AutoProcessor.from_pretrained(processor)
            
        self.zeroshot = zeroshot


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
        speech_features = self.encode_speech(audio_input)
        phone_features = self.encode_phones(phone_input)
        return speech_features, phone_features
    
    def encode_speech(self,audio_input):
        speech_features = self.speech_encoder(**audio_input).pooler_output
        return speech_features
    
    def encode_phones(self, phone_input):
        phone_features = self.phone_encoder(**phone_input).pooler_output
        return phone_features
    
    def training_step(self, batch, batch_idx, **kwargs):
        #print(batch)
        prob = np.random.uniform()
        if prob >= 0.5:
            dataset = 'mswc'
        else:
            dataset = 'fleurs'
        
        if batch['fleurs'] is None:
            dataset = 'mswc'
        if batch['mswc'] is None:
            dataset = 'fleurs'
        
        if self.hard_negatives:
            audio_input, phone_input, negative_input = batch[dataset]

            speech_features, phone_features = self(audio_input, phone_input)

            loss = self.loss(speech_features,
                             phone_features,
                             self.phone_encoder.t_prime,
                             self.phone_encoder.b)
            
            negative_features = self.encode_phones(negative_input)
            
            loss += self.loss(speech_features,
                              negative_features,
                              self.phone_encoder.t_prime,
                              self.phone_encoder.b,
                              only_negatives=True)
            loss = loss/2
        
        else:
            audio_input, phone_input = batch[dataset]

            speech_features, phone_features = self(audio_input, phone_input)

            loss = self.loss(speech_features,
                             phone_features,
                             self.phone_encoder.t_prime,
                             self.phone_encoder.b)

        self.log("loss", loss, prog_bar=True)
        self.log('t_prime',self.phone_encoder.t_prime)
        self.log('b',self.phone_encoder.b)

        return loss


    def validation_step(self, batch, batch_idx, **kwargs):
        #print(batch)
        audio_input, phone_input = batch


        speech_features, phone_features = self(audio_input, phone_input)

        loss = self.loss(speech_features,phone_features,
                             self.phone_encoder.t_prime,
                             self.phone_encoder.b)

        self.validation_step_outputs.append({'val_loss':loss})
        
        

    def on_validation_epoch_end(self):
        
        if self.global_rank == 0:

            avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
            
            self.log("val_loss", avg_loss, sync_dist=True)
        
            if self.zeroshot:
                
                accuracy = Google_Command_Eval(self.speech_encoder, self.phone_encoder, self.tokenizer, self.processor, text=self.text)
                
                self.log("GoogleCommand",accuracy,sync_dist=True)
                
                u_accuracy = UCLA_Phonetic_Corpus_Eval(self.speech_encoder, self.phone_encoder, self.tokenizer, self.processor)
                
                self.log("UCLA",u_accuracy,sync_dist=True)
                




            