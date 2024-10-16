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
	def __init__(self, path):
		self.dictionary = Dictionary()
		self.train = self.tokenize(os.path.join(path, 'train.txt'))
		self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
		self.test = self.tokenize(os.path.join(path, 'test.txt'))
	
	def tokenize(self, path):
		r"""
		Tokenizes a text file.
		"""
		assert os.path.exists(path)
		# 这里的代码似乎可以简化，遂修改（还没有验证正确性）
		with open(path, 'r', encoding="utf8") as f:
			idss = []
			for line in f:
				ids = []
				words = line.split() + ['<eos>']
				for word in words:
					self.dictionary.add_word(word)
					ids.append(self.dictionary.word2idx[word])
				idss.append(torch.tensor(ids).type(torch.int64))
			ids = torch.cat(idss)
		return ids
