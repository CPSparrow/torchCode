"""
    Transformer-Torch
    zufe_ylxn
"""

import math
import time

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import jieba

# S：表示解码输入开始的符号
# E：显示解码输出开始的符号
# P：如果当前批次数据大小短于时间步长，则将填充空白序列的符号


batch_size = 2
device = torch.device("cuda" if torch.cuda.is_available() and True else "cpu")
en_name = './corpus/train/EN.txt'
cn_name = './corpus/train/CN.txt'


def file2tokens(file_name: str):
	r"""
	return lists of words and max len
	"""
	with open(file_name, 'r') as file:
		data = file.read()
		data = data.split('\n')
		data = [list(jieba.cut(t)) for t in data]
		data = [t for t in data if len(t) > 5]
		return data


def tokens2vocab(data, target=None):
	r"""
	根据分词后的文本生成对应的词表(vocab)和相关数据

	:param data:generated by file2tokens
	:param target: should be 'tgt' or 'src'
	:return:
	"""
	if target == 'src':
		vocab = {'P': 0}
		for sentence in data:
			for token in sentence:
				if not token in vocab:
					vocab[token] = len(vocab)
		idx2token = {i: t for i, t in enumerate(vocab)}
		vocab_size = len(vocab)
	elif target == 'tgt':
		vocab = {'P': 0, 'S': 1, 'E': 2}
		for sentence in data:
			for token in sentence:
				if not token in vocab:
					vocab[token] = len(vocab)
		idx2token = {i: t for i, t in enumerate(vocab)}
		vocab_size = len(vocab)
	else:
		raise ValueError("invalid param about target!")
	return vocab, idx2token, vocab_size


def make_data(src_tokens: list, tgt_tokens: list, src_vocab: dict, tgt_vocab: dict):
	r"""
	把分词后的文本转化成下标序列。本函数同时也要实现了padding的功能。
	:param token_lists:输入由file2tokens生成的src_tokens和tgt_tokens
	:param size: 分别需要作padding的长度
	:return data_list: [LongTensor, LongTensor, LongTensor]
	"""
	enc_inputs, dec_inputs, dec_outputs = [], [], []
	
	def pad(x, max_len):
		# 这里的padding的方法可能会有性能上的问题，但就先这样吧。
		x = x + [0] * (max_len - len(x))
		return x
	
	src_len = max([len(sentence) for sentence in src_tokens])
	tgt_len = max([len(sentence) for sentence in tgt_tokens])
	for src in src_tokens:
		src_input = pad([src_vocab[token] for token in src], src_len)
		enc_inputs.append(src_input)
	
	for tgt in tgt_tokens:
		tgt_input = pad([tgt_vocab[token] for token in tgt], tgt_len)
		dec_inputs.append([tgt_vocab['S']] + tgt_input)
		dec_outputs.append(tgt_input + [tgt_vocab['E']])
	
	data_list = [torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)]
	return data_list


src_tokens = file2tokens(en_name)
src2idx, idx2src, src_vocab_size = tokens2vocab(src_tokens, 'src')

tgt_tokens = file2tokens(cn_name)
tgt2idx, idx2tgt, tgt_vocab_size = tokens2vocab(tgt_tokens, 'tgt')

data_list = make_data(src_tokens, tgt_tokens, src2idx, tgt2idx)
src_len, tgt_len = data_list[0].shape[1], data_list[1].shape[1]

# 设置 Transformer 的一些参数
d_model = 512  # Embedding Size 嵌入层
d_ff = 2048  # FeedForward dimension 残差神经网络层
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder and Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention 多头自注意


class MyDataSet(Data.Dataset):
	def __init__(self, data_list):
		super(MyDataSet, self).__init__()
		self.enc_inputs, self.dec_inputs, self.dec_outputs = data_list
	
	def __len__(self):
		return self.enc_inputs.shape[0]
	
	def __getitem__(self, idx):
		return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]


loader = Data.DataLoader(MyDataSet(data_list), batch_size, shuffle=True)


class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		pe = torch.zeros(max_len, d_model)
		# 列矩阵,对应pos,unsqueeze起到了reshape的作用
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		# 行矩阵,对应i
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		# 如果没有下面这句话会怎么样？可以尝试一下
		self.register_buffer('pe', pe)
	
	def forward(self, x):
		"""
		x: [seq_len, batch_size, d_model]
		"""
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
	'''
	seq_q: [batch_size, seq_len]
	seq_k: [batch_size, seq_len]
	seq_len 可以是 src_len 或 tgt_len
	seq_q  seq_k 的 seq_len 可以不相同
	'''
	batch_size, len_q = seq_q.size()
	batch_size, len_k = seq_k.size()
	pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
	return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
	'''
	seq: [batch_size, tgt_len]
	'''
	attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
	subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Upper triangular matrix
	subsequence_mask = torch.from_numpy(subsequence_mask).byte()
	return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class ScaledDotProductAttention(nn.Module):
	def __init__(self):
		super(ScaledDotProductAttention, self).__init__()
	
	def forward(self, Q, K, V, attn_mask):
		"""
		Q: [batch_size, n_heads, len_q, d_k]
		K: [batch_size, n_heads, len_k, d_k]
		V: [batch_size, n_heads, len_v(=len_k), d_v]
		attn_mask: [batch_size, n_heads, seq_len, seq_len]
		"""
		# Self-Attention的经典公式
		# matmul函数用于矩阵的点积
		scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
		scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
		
		attn = nn.Softmax(dim=-1)(scores)
		context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
		return context, attn


class MultiHeadAttention(nn.Module):
	def __init__(self):
		super(MultiHeadAttention, self).__init__()
		self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
		self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
		self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
		self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
	
	def forward(self, input_Q, input_K, input_V, attn_mask):
		'''
		input_Q: [batch_size, len_q, d_model]
		input_K: [batch_size, len_k, d_model]
		input_V: [batch_size, len_v(=len_k), d_model]
		attn_mask: [batch_size, seq_len, seq_len]
		'''
		residual, batch_size = input_Q, input_Q.size(0)
		# (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
		Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
		K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
		V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,
																		   2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
		
		attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1,
												  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
		
		# context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
		context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
		# context: [batch_size, len_q, n_heads * d_v]
		context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
		output = self.fc(context)  # [batch_size, len_q, d_model]
		return nn.LayerNorm(d_model).to(device)(output) + residual.to(device), attn


class PoswiseFeedForwardNet(nn.Module):
	def __init__(self):
		super(PoswiseFeedForwardNet, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(d_model, d_ff, bias=False),
			nn.ReLU(),
			nn.Linear(d_ff, d_model, bias=False)
		)
	
	def forward(self, inputs):
		'''
		inputs: [batch_size, seq_len, d_model]
		'''
		residual = inputs
		output = self.fc(inputs)
		return nn.LayerNorm(d_model).to(device)(output) + residual.to(device)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
	def __init__(self):
		super(EncoderLayer, self).__init__()
		self.enc_self_attn = MultiHeadAttention()
		self.pos_ffn = PoswiseFeedForwardNet()
	
	def forward(self, enc_inputs, enc_self_attn_mask):
		'''
		enc_inputs: [batch_size, src_len, d_model]
		enc_self_attn_mask: [batch_size, src_len, src_len]
		'''
		# enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
		enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
											   enc_self_attn_mask)  # enc_inputs to same Q,K,V
		enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
		return enc_outputs, attn


class DecoderLayer(nn.Module):
	def __init__(self):
		super(DecoderLayer, self).__init__()
		self.dec_self_attn = MultiHeadAttention()
		self.dec_enc_attn = MultiHeadAttention()
		self.pos_ffn = PoswiseFeedForwardNet()
	
	def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
		"""
		dec_inputs: [batch_size, tgt_len, d_model]
		enc_outputs: [batch_size, src_len, d_model]
		dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
		dec_enc_attn_mask: [batch_size, tgt_len, src_len]
		"""
		# dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
		dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
		# dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
		dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
		dec_outputs = self.pos_ffn(dec_outputs)  # [batch_size, tgt_len, d_model]
		return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.src_emb = nn.Embedding(src_vocab_size, d_model)
		self.pos_emb = PositionalEncoding(d_model)
		self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
	
	def forward(self, enc_inputs):
		'''
		enc_inputs: [batch_size, src_len]
		'''
		enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
		enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
		enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
		enc_self_attns = []
		for layer in self.layers:
			# enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
			enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
			enc_self_attns.append(enc_self_attn)
		return enc_outputs, enc_self_attns


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
		self.pos_emb = PositionalEncoding(d_model)
		self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
	
	def forward(self, dec_inputs, enc_inputs, enc_outputs):
		'''
		dec_inputs: [batch_size, tgt_len]
		enc_intpus: [batch_size, src_len]
		enc_outputs: [batsh_size, src_len, d_model]
		'''
		dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
		dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(
			device)  # [batch_size, tgt_len, d_model]
		dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(device)  # [batch_size, tgt_len, tgt_len]
		dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(
			device)  # [batch_size, tgt_len, tgt_len]
		dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
									  0).to(device)  # [batch_size, tgt_len, tgt_len]
		
		dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]
		
		dec_self_attns, dec_enc_attns = [], []
		for layer in self.layers:
			# dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
			dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
															 dec_enc_attn_mask)
			dec_self_attns.append(dec_self_attn)
			dec_enc_attns.append(dec_enc_attn)
		return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
	def __init__(self):
		super(Transformer, self).__init__()
		self.encoder = Encoder().to(device)
		self.decoder = Decoder().to(device)
		self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False).to(device)
	
	def forward(self, enc_inputs, dec_inputs):
		'''
		enc_inputs: [batch_size, src_len]
		dec_inputs: [batch_size, tgt_len]
		'''
		# tensor to store decoder outputs
		# outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
		
		# enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
		enc_outputs, enc_self_attns = self.encoder(enc_inputs)
		# dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
		dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
		dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
		return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


if __name__ == "__main__":
	model = Transformer().to(device)
	criterion = nn.CrossEntropyLoss(ignore_index=0)
	optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.98)
	
	print("Training...")
	loss = 0.0
	loss_store = []
	
	
	def draw_loss(data):
		import matplotlib.pyplot as plt
		x = [i for i in range(1, len(data) + 1)]
		plt.figure(dpi=160)
		plt.plot(x, data)
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.savefig("loss_fig.png", dpi=300)
	
	
	epoches = 40
	start = time.time()
	for epoch in range(1, epoches) if device == torch.device('cuda') else range(1, 10):
		for enc_inputs, dec_inputs, dec_outputs in loader:
			"""
			enc_inputs: [batch_size, src_len]
			dec_inputs: [batch_size, tgt_len]
			dec_outputs: [batch_size, tgt_len]
			"""
			# 这里一模一样的变量为什么要再复制一次?我也不知道
			enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(device), dec_inputs.to(device), dec_outputs.to(device)
			outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
			loss = criterion(outputs, dec_outputs.view(-1))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			loss_store.append(float(loss))
		if epoch % 10 == 0 or False:
			print('Epoch:', '%04d' % (epoch), 'loss =', '{:.3f}'.format(loss))
	print("Training finished at loss = {} after {} epoches, {:.3f}s used.\n".format(
		round(float(loss), 3), epoches, time.time() - start))
	# print(device)
	draw_loss(loss_store)
	
	
	def greedy_decoder(model, enc_input, start_symbol: int):
		"""
		为了简单起见,当K=1时,贪婪解码器是波束搜索。这对于推理是必要的,因为我们不知道
		目标序列输入。因此,我们尝试逐字生成目标输入，然后将其输入到转换器中。
		param model:tansformer model
		param enc_input:encoder输入
		param start_symbol:开始符号。在本例中,它是“S”,对应于索引4
		return:目标输入
		"""
		enc_outputs, enc_self_attns = model.encoder(enc_input)
		dec_input = torch.zeros(1, 0).type_as(enc_input.data)
		terminal = False
		next_symbol = start_symbol
		cnt = 0
		# 这里设置一个cnt计数单纯是避免在少量训练时输出无法停止
		while not terminal and (cnt <= tgt_len):
			cnt += 1
			dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol]], dtype=enc_input.dtype).to(device)],
								  -1)
			dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
			projected = model.projection(dec_outputs)
			prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
			next_word = prob.data[-1]
			next_symbol = next_word
			if next_symbol == tgt2idx['E']:
				terminal = True
		return dec_input
	
	
	# 测试
	while True:
		enc_inputs, _, _ = next(iter(loader))
		enc_inputs = enc_inputs.to(device)
		for i in range(len(enc_inputs)):
			greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol=tgt2idx["S"])
			predict, _, _, _ = model(enc_inputs[i].view(1, -1), greedy_dec_input)
			predict = predict.data.max(1, keepdim=True)[1]
			# print(enc_inputs[i], '->', [idx2word[n.item()] for n in predict.squeeze()])
			print(repr(
				''.join([idx2src[n.item()] for n in enc_inputs[i].squeeze()]))
				, '->\n',
				repr(
					''.join([idx2tgt[n.item()] for n in predict.squeeze()])))
			print('\n')
		q = input("input q to quit:")
		if q == 'q':
			break
