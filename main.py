import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import false

batch_size = 2

# 关于word embedding，以序列建模为例子

# 单词表大小
max_src_word_nums = 8
max_tgt_word_nums = 8

# embedding 的维度
model_dim = 8

torch_test = false
if torch_test:
    src_len = torch.randint(2, 5, (batch_size,))
    tgt_len = torch.randint(2, 5, (batch_size,))
else:
    src_len = torch.Tensor([2, 4]).to(torch.int32)
    tgt_len = torch.Tensor([4, 3]).to(torch.int32)

# 设定(也可以是求取)序列的最大长度
max_src_len = 5
max_tgt_len = 5
max_position_len = 5

# 单词索引构成的句子,构建了batch,并且作了padding(默认值为0)
src_seq = torch.cat(
    [torch.unsqueeze(F.pad(torch.randint(1, max_src_word_nums, (L,)), (0, max_src_len - L)), 0) \
     for L in src_len])
tgt_seq = torch.cat(
    [torch.unsqueeze(F.pad(torch.randint(1, max_tgt_word_nums, (L,)), (0, max_tgt_len - L)), 0) \
     for L in tgt_len])

# 构造word embedding
src_embedding_table = nn.Embedding(max_src_word_nums + 1, model_dim)
tgt_embedding_table = nn.Embedding(max_tgt_word_nums + 1, model_dim)
src_word_embedding = src_embedding_table(src_seq)
tgt_word_embedding = tgt_embedding_table(tgt_seq)

# 构造position embedding:
# 列矩阵pos
pos_mat = torch.arange(max_position_len).reshape(-1, 1)
# 行矩阵i
i_mat = torch.pow(10000, torch.arange(0, model_dim, 2).reshape(1, -1) / model_dim)

pos_embedding_table = torch.zeros(max_position_len, model_dim)
pos_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
pos_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)

# 完成pos embedding 的构造
pos_embedding = nn.Embedding(max_position_len, model_dim)
pos_embedding.weight = nn.Parameter(pos_embedding_table, requires_grad=False)

# 构造传入序列的位置索引，得到对应的pos embedding
src_pos = torch.cat([torch.unsqueeze(torch.arange(max_src_len), 0) for _ in src_len]).to(torch.int32)
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max_tgt_len), 0) for _ in tgt_len]).to(torch.int32)

src_pos_embedding = pos_embedding(src_pos)
tgt_pos_embedding = pos_embedding(tgt_pos)

# print(repr(src_embedding_table.weight))
print(repr(src_pos_embedding))
print(repr(tgt_pos_embedding))
