import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

batch_size = 2

# 关于word embedding，以序列建模为例子

# 单词表大小
max_src_word_nums = 8
max_tgt_word_nums = 8

# embedding 的维度
model_dim = 8

is_str_len_rand = False
if is_str_len_rand:
    src_len = torch.randint(2, 5, (batch_size,))
    tgt_len = torch.randint(2, 5, (batch_size,))
else:
    src_len = torch.Tensor([2, 4]).to(torch.int32)
    tgt_len = torch.Tensor([4, 3]).to(torch.int32)

# 设定(也可以是求取)序列的最大长度
max_src_len = 5
max_tgt_len = 5
max_position_len = 5

# Step1: 单词索引构成的句子,构建了batch,并且作了padding(默认值为0)
src_seq = torch.cat(
    [torch.unsqueeze(F.pad(torch.randint(1, max_src_word_nums, (L,)), (0, max_src_len - L)), 0) \
     for L in src_len])
tgt_seq = torch.cat(
    [torch.unsqueeze(F.pad(torch.randint(1, max_tgt_word_nums, (L,)), (0, max_tgt_len - L)), 0) \
     for L in tgt_len])

# Step2: 构造word embedding
src_embedding_table = nn.Embedding(max_src_word_nums + 1, model_dim)
tgt_embedding_table = nn.Embedding(max_tgt_word_nums + 1, model_dim)
src_word_embedding = src_embedding_table(src_seq)
tgt_word_embedding = tgt_embedding_table(tgt_seq)

# Step3: 构造position embedding:
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

# Step4: 构造encoder的self-attention mask
# mask 的shape:[batch_size, max_src_len, max_src_len],值为1或-inf
valid_encoder_pos = torch.unsqueeze(
    torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max_src_len - L)), 0) for L in src_len]), 2)
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_encoder_pos_matrix = 1 - valid_encoder_pos_matrix
mask_self_attention = invalid_encoder_pos_matrix.to(torch.bool)

# 目前并不清楚为何要使用随机生成的这个score。
is_score_set = True
if is_score_set:
    score = torch.randn(batch_size, max_src_len, max_src_len)
else:
    # 这里应该是根据src_word_embedding和src_pos_embedding通过某种计算得到的score(这个想法对吗？)
    # 这里尝试给出了score的表达式。矩阵的shape是没错的，但是并不确定是否正确。
    score = torch.bmm(src_word_embedding, src_pos_embedding.transpose(1, 2))
masked_score = score.masked_fill(mask=mask_self_attention, value=-1e9)
prob = F.softmax(masked_score, dim=-1)
# print("{0}\n{1}".format(score, prob))

# Step5: 构造intro_attention的mask
# Q @ K^T shape: [batch_size, max_tgt_len, max_src_len]

# just like valid_encoder_pos
valid_decoder_pos = torch.unsqueeze(
    torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max_tgt_len - L)), 0) for L in tgt_len]), 2)
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_cross_pos_matrix = 1 - valid_cross_pos_matrix
mask_cross_attention = invalid_cross_pos_matrix.to(torch.bool)

# 在这之后要做的事情和前面差不多,mask_fill一下用于完成，代码就不写了

# Step6: 构造decoder self-attention的mask
valid_tri_matrix = [F.pad(torch.tril(torch.ones(L, L)), (0, (max_tgt_len - L), 0, (max_tgt_len - L))) for L in tgt_len]
print("{0}".format(valid_tri_matrix))
