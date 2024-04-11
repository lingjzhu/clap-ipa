import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from numba import jit
from torch.autograd import Function


def ContrastiveSigmoid(s_emb,p_emb,t_prime,b,only_negatives=False):
    # s_emb : speech model embedding [n, dim]
    # p_emb : phoneme model embedding [n, dim]
    # t_prime, b : learnable temperature and bias
    # n : mini-batch size
    n = s_emb.size(0)
    n = torch.tensor(n).to(b.device)
    
    logSigmoid = nn.LogSigmoid()
    t = torch.exp(t_prime)
    z_s = F.normalize(s_emb,dim=-1)
    z_p = F.normalize(p_emb,dim=-1)
    logits = torch.matmul(z_s, z_p.T) * t + b
    
    if only_negatives:
        labels = torch.zeros(n,n) - torch.ones(n) # all -1 
        labels = labels.to(b.device)
    else:
        labels = 2 * torch.eye(n) - torch.ones(n) # -1 with diagonal 1
        labels = labels.to(b.device)

    loss = -torch.sum(logSigmoid(labels * logits)) / n
    return loss



class ForwardSumLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(ForwardSumLoss, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.blank_logprob = blank_logprob
#        self.off_diag_penalty = off_diag_penalty
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)
        
    def forward(self, attn_logprob, text_lens, mel_lens):
        """
        Args:
        attn_logprob: batch x 1 x max(mel_lens) x max(text_lens)
        batched tensor of attention log
        probabilities, padded to length
        of longest sequence in each dimension
        text_lens: batch-D vector of length of
        each text sequence
        mel_lens: batch-D vector of length of
        each mel sequence
        """
        # The CTC loss module assumes the existence of a blank token
        # that can be optionally inserted anywhere in the sequence for
        # a fixed probability.
        # A row must be added to the attention matrix to account for this
        attn_logprob_pd = F.pad(input=attn_logprob,
                                pad=(1, 0, 0, 0, 0, 0, 0, 0),
                                value=self.blank_logprob)
        cost_total = 0.0
        # for-loop over batch because of variable-length
        # sequences
        for bid in range(attn_logprob.shape[0]):
            # construct the target sequence. Every
            # text token is mapped to a unique sequence number,
            # thereby ensuring the monotonicity constraint
            target_seq = torch.arange(1, text_lens[bid]+1)
            target_seq=target_seq.unsqueeze(0)
            curr_logprob = attn_logprob_pd[bid].permute(1, 0, 2)
            curr_logprob = curr_logprob[:mel_lens[bid],:,:text_lens[bid]+1]
            
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            cost = self.CTCLoss(curr_logprob,
                                target_seq,
                                input_lengths=mel_lens[bid:bid+1],
                                target_lengths=text_lens[bid:bid+1])
            cost_total += cost
        # average cost over batch
        cost_total = cost_total/attn_logprob.shape[0]
        return cost_total

class ForwardSumLossWithBlank(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super(ForwardSumLossWithBlank, self).__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        #self.blank_logprob = blank_logprob
#        self.off_diag_penalty = off_diag_penalty
        self.CTCLoss = nn.CTCLoss(zero_infinity=True)
        
    def forward(self, attn_logprob, text_lens, mel_lens):
        """
        Args:
        attn_logprob: batch x 1 x max(mel_lens) x max(text_lens)
        batched tensor of attention log
        probabilities, padded to length
        of longest sequence in each dimension
        text_lens: batch-D vector of length of
        each text sequence
        mel_lens: batch-D vector of length of
        each mel sequence
        """
        # The CTC loss module assumes the existence of a blank token
        # that can be optionally inserted anywhere in the sequence for
        # a fixed probability.
        # A row must be added to the attention matrix to account for this
        #attn_logprob_pd = F.pad(input=attn_logprob,
        #                        pad=(1, 0, 0, 0, 0, 0, 0, 0),
        #                        value=self.blank_logprob)
        cost_total = 0.0
        # for-loop over batch because of variable-length
        # sequences
        for bid in range(attn_logprob.shape[0]):
            # construct the target sequence. Every
            # text token is mapped to a unique sequence number,
            # thereby ensuring the monotonicity constraint
            target_seq = torch.arange(text_lens[bid])
            target_seq=target_seq.unsqueeze(0)
            curr_logprob = attn_logprob[bid].permute(1, 0, 2)
            curr_logprob = curr_logprob[:mel_lens[bid],:,:text_lens[bid]]
            
            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            cost = self.CTCLoss(curr_logprob,
                                target_seq,
                                input_lengths=mel_lens[bid:bid+1],
                                target_lengths=text_lens[bid:bid+1])
            cost_total += cost
        # average cost over batch
        cost_total = cost_total/attn_logprob.shape[0]
        return cost_total
    

def _entropy(logits: torch.Tensor) -> torch.Tensor:
    r"""Compute entropy according to the definition.

    Args:
        logits: Unscaled log probabilities.

    Return:
        A tensor containing the Shannon entropy in the last dimension.
    """
    probs = F.softmax(logits, dim=-1) + 1e-8
    entropy = - probs * torch.log(probs)
    entropy = torch.sum(entropy, dim=-1)
    return entropy.mean()
    
def entropy_loss(logits: torch.Tensor, phone_len: torch.Tensor,speech_len: torch.Tensor) -> torch.Tensor:
    
    
    B = logits.size(0)
    losses = torch.zeros(B)
    for i in range(B):
        loss = _entropy(logits[i][:speech_len[i],:phone_len[i]])
        losses[i] = loss
    return torch.mean(losses)



        
    

if __name__ == "__main__":
    
    pass