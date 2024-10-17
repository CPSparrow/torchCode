import json
import os
import torch
from io import open
from jieba import cut


class Dictionary(object):
	def __init__(self):
		self.word2idx = {}
		self.idx2word = []
	
	def add_word(self, word):
		if word not in self.word2idx:
			self.idx2word.append(word)
			self.word2idx[word] = len(self.idx2word) - 1
		# 提问：为什么要返回呢？
		return self.word2idx[word]
	
	def __len__(self):
		return len(self.idx2word)


class Corpus(object):
	def __init__(self, path, len_seq):
		self.dictionary = Dictionary()
		self.train, self.valid, self.test = (
			self.tokenize(os.path.join(path, 'raw.txt'), len_seq))
	
	def tokenize(self, path, len_seq=80):
		r"""
		Tokenizes a text file.
		"""
		assert os.path.exists(path)
		batched_src = []
		with open(path, 'r', encoding="utf8") as src_file:
			store = []
			for line in src_file:
				store.extend([i for i in cut(line, cut_all=False) if i != '\n'])
				while len(store) > len_seq:
					batched_src.append(store[0:len_seq - 1] + ['<eos>'])
					store = store[len_seq - 1:]
		
		train, valid, test = [], [], []
		for interval, batch in enumerate(batched_src):
			idx_list = []
			for word in batch:
				self.dictionary.add_word(word)
				idx_list.append(self.dictionary.word2idx[word])
			tensor = torch.tensor(idx_list, dtype=torch.int64)
			if interval % 10 < 7:
				train.append(tensor)
			elif interval % 10 >= 9:
				test.append(tensor)
			else:
				valid.append(tensor)
		return torch.cat(train), torch.cat(valid), torch.cat(test)


if __name__ == "__main__":
	with open("./utils/cfg.json") as cfg_file:
		cfg = json.load(cfg_file)
	corpus = Corpus(cfg["data"], cfg["len_seq"])
	print(len(corpus.train))
	print(len(corpus.test))
	print(len(corpus.valid))
	print(round(len(corpus.train) / cfg["batch_size"], 2))
