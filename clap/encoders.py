from transformers import BertModel, BertTokenizer, BertConfig, WhisperModel, WhisperConfig
from transformers import BertPreTrainedModel,WhisperPreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPoolingAndCrossAttentions
)
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from clap.WhisperEncoder import WhisperEncoder



def MeanPooler(hidden_states,attention_mask):
    if attention_mask is not None:
        pooled_output = torch.sum(hidden_states*attention_mask.unsqueeze(-1),dim=1)/torch.sum(attention_mask,dim=-1).unsqueeze(-1) 
    else:
        pooled_output = torch.mean(hidden_states,dim=1)
    return pooled_output



class PhoneEncoder(BertPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.projector = nn.Linear(config.hidden_size, config.proj_size)
        self.pooler = MeanPooler
        
        if config.learnable_scale:
            self.t_prime = nn.Parameter(torch.tensor(config.t_prime).float())
            self.b = nn.Parameter(torch.tensor(config.b).float())
        else:
            self.t_prime = nn.Parameter(torch.tensor(config.t_prime).float(),requires_grad=False)
            self.b = nn.Parameter(torch.tensor(config.b).float(),requires_grad=False)

        # Initialize weights and apply final processing
        self.post_init()    
    

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = self.projector(outputs.last_hidden_state)
        pooled_output = self.pooler(sequence_output,attention_mask)


        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=None
        )
    
    
    
class SpeechEncoder(WhisperPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)

        self.projector = nn.Linear(config.hidden_size, config.proj_size)
        self.pooler = MeanPooler

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = self.projector(encoder_outputs.last_hidden_state)
        pooled_output = self.pooler(hidden_states,attention_mask[:,::2])


        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=None,
        )
    
    

def initialize_phone_model(cfg):
    
    config = BertConfig.from_pretrained(cfg['name'])
    config.proj_size = cfg['projection_size']
    config.t_prime = cfg['t_prime']
    config.b = cfg['b']
    config.learnable_scale = cfg.get('learnable_scale',True)
    if not cfg['pretrained']:
        model = PhoneEncoder(config)
    else:
        model = PhoneEncoder.from_pretrained(cfg['name'],config=config)
                                          
    
    if cfg['gradient_checkpointing']:
        model._set_gradient_checkpointing(model.bert, True)
    
    return model


def initialize_speech_model(cfg):
    
    config = WhisperConfig.from_pretrained(cfg['name'])
    config.apply_spec_augment = cfg['apply_spec_augment']
    config.proj_size = cfg['projection_size']
    model = SpeechEncoder.from_pretrained(cfg['name'],config=config)
    
    if cfg['gradient_checkpointing']:
        model.encoder.gradient_checkpointing = True
        
    return model
    


if __name__ == "__main__":
    
    pass