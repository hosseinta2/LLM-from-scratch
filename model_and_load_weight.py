#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 23:09:19 2026

@author: hosseintaheri
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GELU(nn.Module): # efficient GELU activation function implementation
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *(x + 0.044715 * torch.pow(x, 3))))
    
class feedforwardNN(nn.Module): # 1-hidden-layer network with GELU act. function
  def __init__(self,emb_dim):
     super().__init__()
     self.layers = nn.Sequential(nn.Linear(emb_dim,4*emb_dim),GELU(),
                                 nn.Linear(4*emb_dim,emb_dim))
  def forward(self,x):
    return self.layers(x)

class norm(nn.Module): #normalization layer, input size = batch * context_size * d_embed
  def __init__(self, emb_dim):
     super().__init__()
     self.eps = 1e-5
     self.scale = nn.Parameter(torch.ones(emb_dim)) # trainable parameters for better training stability, initialized as one for scaling and zero for shifting
     self.shift = nn.Parameter(torch.zeros(emb_dim))
  def forward(self, x):
    mean = x.mean(dim=-1,keepdim=True) # average along the d_embed dimension
    variance = x.var(dim=-1,keepdim=True, unbiased=False)
    normalized_x = (x-mean)/torch.sqrt(variance+self.eps)
    return self.scale*normalized_x + self.shift


class MHA_v2(nn.Module): # multi-head attention, input size = batch * context size * d_embed (input embedding dimension),
                         #output size = batch * context size * d_embed (attention embedding dimension, usually equal to input embedding dimension)
  def __init__(self,d_in,d_out,b,do_rate,num_heads,context_size):
     super().__init__()
     assert d_out%num_heads==0
     self.head_dim = d_out//num_heads # embedding dimension for each head
     self.num_heads = num_heads
     self.W_k = nn.Linear(d_in,d_out,bias=b) # W_key
     self.W_q = nn.Linear(d_in,d_out,bias=b) # W_query
     self.W_v = nn.Linear(d_in,d_out,bias=b) # W_value \in R^{d_{embedding}* d_{embedding}}
     self.drop_out= nn.Dropout(do_rate)
     self.register_buffer('mask',torch.triu(torch.ones(context_size,context_size),1)) # Cuausal Mask
     self.projection = nn.Linear(d_out,d_out) # projection for outer layer (Optional)
  def forward(self,x):
    X_k = self.W_k(x) # X_k = X*W_k \in R^{batches*context_size*d_embed}
    X_q = self.W_q(x) #       X*W_q
    X_v = self.W_v(x) #       X*W_v
    batch_size,context_size,d_out = X_k.shape
    head_dim = self.head_dim
    X_k = X_k.view(batch_size, context_size, self.num_heads, head_dim) # partition X_k into different heads
    X_q = X_q.view(batch_size, context_size, self.num_heads, head_dim)
    X_v = X_v.view(batch_size, context_size, self.num_heads, head_dim)
    X_k = X_k.transpose(1,2) # transpose dimensions context_size and self.num_heads to prepare the matrices for per batch and per head multiplication
    X_q = X_q.transpose(1,2)
    X_v = X_v.transpose(1,2)
    atten_prod = X_q@X_k.transpose(2,3) # for each head and batch, attention weights = X_q * X_k'
    atten_prod = atten_prod.masked_fill(self.mask.bool()[:context_size,:context_size],-torch.inf) # causal mask
    atten_scores = torch.softmax(atten_prod/X_k.shape[-1]**0.5,dim=-1) # softmax along the columns, so each row of atten_scores[batch,heads,:,:] is a probability vector.
    atten_scores = self.drop_out(atten_scores) # applt dropout on attention matrix for each batch and head
    output = atten_scores@X_v # output size = batch*num_heads*context_size*head_head_dim
    output = output.transpose(1,2) # output size = batch*context_size*num_heads*head_head_dim
    d_out= self.num_heads*head_dim
    output = output.contiguous().view(batch_size,context_size,d_out)# concatenation of all heads, contiguous required since transpose was used on output before,
    output = self.projection(output) # optional, projection layer
    return output



GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,      # Context length (originally 1024 for gpt2)
"emb_dim": 768,
"n_heads": 12,
"n_layers": 12,
"drop_rate": 0.1,
"qkv_bias": False
# Embedding dimension
# Number of attention heads
# Number of layers
# Dropout rate
# Query-Key-Value bias
}

class transformerblock(nn.Module): # one-layer transformer block with MHA, FFN and residual connections, x size = [batch_size, context size, d_embedding]
  def __init__(self, cg):
     super().__init__()
     self.mha = MHA_v2(cg["emb_dim"],cg["emb_dim"],cg["qkv_bias"],cg["drop_rate"],
                       cg["n_heads"],cg["context_length"])
     self.norm1 = norm(cg["emb_dim"])
     self.norm2 = norm(cg["emb_dim"])
     self.ff = feedforwardNN(cg["emb_dim"])
     self.drop_res = nn.Dropout(cg["drop_rate"])
  def forward(self,x):
    shortcut = x
    x = self.norm1(x) # normalization layer for each token in each batch. nowadays it is moved after the residual connection
    x = self.mha(x)
    x = self.drop_res(x)
    x = x + shortcut

    shortcut= x # normalization layer for each token in each batch. nowadays it is moved after the residual connection
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_res(x)
    x = shortcut+x
    return x


class GPT(nn.Module): # full GPT model with input size = [batch_size,context size] and each element is in {0,1,...,vocab_size}
  def __init__(self,cg):
    super().__init__()
    self.token_emb = nn.Embedding(cg["vocab_size"],cg["emb_dim"]) # token embeddings, for each row (among vocab_size rows), outputs a d_embedding vector
    self.pos_emb = nn.Embedding(cg["context_length"],cg["emb_dim"]) # positional embedding, for each row (among context_size rows) outputs a d_embedding dimensional vector
    self.drop_emb = nn.Dropout(cg["drop_rate"])
    self.trans_blocks = nn.Sequential(*[transformerblock(cg) for _ in range(cg["n_layers"])]) # sequential layers of multi-head transformer block
    self.final_norm = norm(cg["emb_dim"]) # normalization before output
    self.out_proj = nn.Linear(cg["emb_dim"],cg["vocab_size"],bias=False) # output projection transforming [batch * context_size * d_embedding] to [batch*context_size*vocab_size]

  def forward(self, x):
    batch_size, num_tokens = x.shape
    x = self.token_emb(x)
    #x size is now [batch_size, context_size,d_embbedding]
    positions = self.pos_emb(torch.arange(num_tokens,device = x.device))
    x = x+positions
    x = self.drop_emb(x)
    x = self.trans_blocks(x)
    x = self.final_norm(x)
    logits = self.out_proj(x) # output is [num_batches,context_size,vocab_size]
    return logits




model = GPT(GPT_CONFIG_124M)




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
model.to(device)




import numpy as np # loading the weights into our GPT model

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])  #A
    gpt.token_emb.weight = assign(gpt.token_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):  #B
        q_w, k_w, v_w = np.split(  #C
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trans_blocks[b].mha.W_q.weight = assign(
            gpt.trans_blocks[b].mha.W_q.weight, q_w.T)
        gpt.trans_blocks[b].mha.W_k.weight = assign(
            gpt.trans_blocks[b].mha.W_k.weight, k_w.T)
        gpt.trans_blocks[b].mha.W_v.weight = assign(
            gpt.trans_blocks[b].mha.W_v.weight, v_w.T)
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trans_blocks[b].mha.W_q.bias = assign(
            gpt.trans_blocks[b].mha.W_q.bias, q_b)
        gpt.trans_blocks[b].mha.W_k.bias = assign(
            gpt.trans_blocks[b].mha.W_k.bias, k_b)
        gpt.trans_blocks[b].mha.W_v.bias = assign(
            gpt.trans_blocks[b].mha.W_v.bias, v_b)
        gpt.trans_blocks[b].mha.projection.weight = assign(
            gpt.trans_blocks[b].mha.projection.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trans_blocks[b].mha.projection.bias = assign(
            gpt.trans_blocks[b].mha.projection.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trans_blocks[b].ff.layers[0].weight = assign(
            gpt.trans_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trans_blocks[b].ff.layers[0].bias = assign(
            gpt.trans_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trans_blocks[b].ff.layers[2].weight = assign(
            gpt.trans_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trans_blocks[b].ff.layers[2].bias = assign(
            gpt.trans_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trans_blocks[b].norm1.scale = assign(
            gpt.trans_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trans_blocks[b].norm1.shift = assign(
            gpt.trans_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trans_blocks[b].norm2.scale = assign(
            gpt.trans_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trans_blocks[b].norm2.shift = assign(
            gpt.trans_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_proj.weight = assign(gpt.out_proj.weight, params["wte"])  #D
