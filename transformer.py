#!/usr/bin/python
# -*- coding: UTF-8 -*-

import copy
import logging
import math

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import constants as ct


HIDDEN_SIZE = ct.HIDDEN_SIZE
INPUT_FEATURES_DIMS = ct.INPUT_FEATURES_DIMS
HEADS = ct.HEADS
GRID_SIZE = ct.GRID_SIZE
DOWNSAMPLING_RATE = ct.DOWNSAMPLING_RATE
ENCODER_LAYERS = ct.ENCODER_LAYERS

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, ):
        super(Multi_Head_Self_Attention, self).__init__()
        self.num_attention_heads = HEADS
        self.hidden_size = HIDDEN_SIZE
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(self.hidden_size, self.all_head_size)
        self.key = Linear(self.hidden_size, self.all_head_size)
        self.value = Linear(self.hidden_size, self.all_head_size)

        self.out = Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = Dropout(0)
        self.proj_dropout = Dropout(0)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states ):
        mixed_query_layer = self.query(hidden_states) 
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        attention_output = context_layer
        return attention_output


class Mlp(nn.Module):
    def __init__(self,):
        super(Mlp, self).__init__()
        self.num_attention_heads = HEADS
        self.hidden_size = HIDDEN_SIZE
        self.mlp_dim = 2048
        self.fc1 = Linear(self.hidden_size, self.mlp_dim)
        self.fc2 = Linear(self.mlp_dim, self.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Encoder_Block(nn.Module):
    def __init__(self,):
        super(Encoder_Block, self).__init__()
        self.num_attention_heads = HEADS
        self.hidden_size = HIDDEN_SIZE
        self.attention_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.ffn = Mlp()
        self.attn = Multi_Head_Self_Attention()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x



class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(self.hidden_size, eps=1e-6)
        for _ in range(ENCODER_LAYERS):
            layer = Encoder_Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded
        
class Embeddings(nn.Module):
    """Construct the embeddings from extracted features or raw images and assign position embeddings.
    """
    def __init__(self, img_size=256, in_channels=INPUT_FEATURES_DIMS):
        super(Embeddings, self).__init__()
        self.hidden_size = HIDDEN_SIZE
        img_size = _pair(img_size)
        grid_size = GRID_SIZE
        ds_rate = DOWNSAMPLING_RATE
        patch_size = (img_size[0] // ds_rate // grid_size, img_size[1] // ds_rate // grid_size)
        patch_size_on_image = (patch_size[0] * ds_rate, patch_size[1] * ds_rate)
        n_patches = (img_size[0] // patch_size_on_image[0]) * (img_size[1] // patch_size_on_image[1])  

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=self.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, self.hidden_size))

        self.dropout = Dropout(0.2)


    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
          
class Transformer(nn.Module):
    def __init__(self, ):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings()
        self.encoder = Encoder()

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        img_features = embedding_output
        encoded = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded

def main():
    net = Transformer()
    y = net(torch.randn(4,1024,16,16))
    print('y_outputs', y.shape)  
    
if __name__ == '__main__':
    main()
