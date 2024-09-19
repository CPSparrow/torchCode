# 这代码真的没有问题吗？输出的 q k score居然是一样的

import numpy as np
import torch
from torch import Tensor
from typing import Optional, Any, Union, Callable
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time


class MultiHeadedAttention(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # Assume v_dim always equals k_dim
        self.k_dim = d_model // num_heads
        self.num_heads = num_heads
        self.proj_weights = clones(nn.Linear(d_model, d_model), 4)  # W^Q, W^K, W^V, W^O
        self.attention_score = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[Tensor] = None):
        """
        Args:
            query: shape (batch_size, seq_len, d_model)
            key: shape (batch_size, seq_len, d_model)
            value: shape (batch_size, seq_len, d_model)
            mask: shape (batch_size, seq_len, seq_len). Since we assume all data use a same mask, so
                  here the shape also equals to (1, seq_len, seq_len)

        Return:
            out: shape (batch_size, seq_len, d_model). The output of a multihead attention layer
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # 1) Apply W^Q, W^K, W^V to generate new query, key, value
        query, key, value \
            = [proj_weight(x).view(batch_size, -1, self.num_heads, self.k_dim).transpose(1, 2)
               for proj_weight, x in zip(self.proj_weights, [query, key, value])]  # -1 equals to seq_len

        # 2) Calculate attention score and the out
        out, self.attention_score = attention(query, key, value, mask=mask,
                                              dropout=self.dropout)

        # 3) "Concat" output
        out = out.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.num_heads * self.k_dim)

        # 4) Apply W^O to get the final output
        out = self.proj_weights[-1](out)

        return out


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query: Tensor,
              key: Tensor,
              value: Tensor,
              mask: Optional[Tensor] = None,
              dropout: float = 0.1):
    """
    Define how to calculate attention score
    Args:
        query: shape (batch_size, num_heads, seq_len, k_dim)
        key: shape(batch_size, num_heads, seq_len, k_dim)
        value: shape(batch_size, num_heads, seq_len, v_dim)
        mask: shape (batch_size, num_heads, seq_len, seq_len). Since our assumption, here the shape is
              (1, 1, seq_len, seq_len)
    Return:
        out: shape (batch_size, v_dim). Output of an attention head.
        attention_score: shape (seq_len, seq_len).

    """
    k_dim = query.size(-1)

    # shape (seq_len ,seq_len)，row: token，col: that token's attention score
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(k_dim)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)

    attention_score = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_score = dropout(attention_score)

    out = torch.matmul(attention_score, value)
    print(value.shape, '  ', scores.shape, '\n',
          attention_score.shape, '  ', out.shape, '\n')

    return out, attention_score  # shape: (seq_len, v_dim), (seq_len, seq_lem)


if __name__ == '__main__':
    d_model = 512
    seq_len = 1024
    batch_size = 32
    num_heads = 8
    # mask = None
    mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0).unsqueeze(0)

    input_matrix = torch.rand(batch_size, seq_len, d_model)
    multi_attn = MultiHeadedAttention(num_heads=num_heads, d_model=d_model, dropout=0.1)
    out = multi_attn(query=input_matrix, key=input_matrix, value=input_matrix, mask=mask)
    print(out.shape)
