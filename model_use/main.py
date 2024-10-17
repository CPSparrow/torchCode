# coding: utf-8
import argparse
import json
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx

import data
import model
from torch import autocast


def batchify(raw_data, bsz):
	"""
	Starting from sequential data, batchify arranges the dataset into columns.
	For instance, with the alphabet as the sequence and batch size 4, we'd get
	┌ a g m s ┐
	│ b h n t │
	│ c i o u │
	│ d j p v │
	│ e k q w │
	└ f l r x ┘.
	These columns are treated as independent by the model, which means that the
	dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
	batch processing.
	"""
	n_batch = raw_data.size(0) // bsz
	raw_data = raw_data.narrow(0, 0, n_batch * bsz)
	raw_data = raw_data.view(bsz, -1).t().contiguous()
	return raw_data.to(device)


def get_batch(source, i):
	"""
	get_batch subdivides the source data into chunks of length len_seq.
	If source is equal to the example output of the batchify function, with
	a len_seq-limit of 2, we'd get the following two Variables for i = 0:
	┌ a g m s ┐ ┌ b h n t ┐
	└ b h n t ┘ └ c i o u ┘
	Note that despite the name of the function, the subdivison of data is not
	done along the batch dimension (i.e. dimension 1), since that was handled
	by the batchify function. The chunks are along dimension 0, corresponding
	to the seq_len dimension in the LSTM.
	"""
	seq_len = min(len_seq, len(source) - 1 - i)
	src = source[i:i + seq_len]
	tgt = source[i + 1:i + 1 + seq_len].view(-1)
	return src, tgt


def evaluate(data_source):
	# Turn on evaluation mode which disables dropout.
	model.eval()
	total_loss = 0.
	with torch.no_grad():
		for i in range(0, data_source.size(0) - 1, len_seq):
			src, tgt = get_batch(data_source, i)
			if cfg["model_type"] == 'Transformer':
				output = model(src)
				output = output.view(-1, vocab_size)
			# .item()的作用是把size为1的tensor转化为标量
			total_loss += len(src) * criterion(output, tgt).item()
	return total_loss / (len(data_source) - 1)


def train():
	# Turn on training mode which enables dropout.
	model.train()
	total_loss = 0.
	start_time = time.time()
	for batch, i in enumerate(range(0, train_data.size(0) - 1, len_seq)):
		src, tgt = get_batch(train_data, i)
		model.zero_grad()
		assert cfg["model_type"] == 'Transformer'
		output = model(src)
		output = output.view(-1, vocab_size)
		loss = criterion(output, tgt)
		loss.backward()
		
		torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
		for p in model.parameters():
			p.data.add_(p.grad, alpha=-lr)
		
		total_loss += loss.item()
		
		if batch % cfg["train_log_interval"] == 0 and batch > 0:
			cur_loss = total_loss / cfg["train_log_interval"]
			elapsed = time.time() - start_time
			print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
				  'loss {:5.2f} | ppl {:8.2f}'.format(
				epoch, batch, len(train_data) // len_seq, lr,
							  elapsed * 1000 / cfg["train_log_interval"], cur_loss, math.exp(cur_loss)))
			total_loss = 0
			start_time = time.time()


def export_onnx(path, model_batch_size, seq_len):
	print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(cfg["onnx_path"])))
	model.eval()
	dummy_input = torch.LongTensor(seq_len * model_batch_size).zero_().view(-1, model_batch_size).to(device)
	hidden = model.init_hidden(model_batch_size)
	torch.onnx.export(model, (dummy_input, hidden), path)


if __name__ == "__main__":
	with open("./utils/cfg.json") as cfg_file:
		cfg = json.load(cfg_file)
	
	# Set the random seed manually for reproducibility.
	torch.manual_seed(cfg["seed"])
	
	device = torch.device("cuda") if \
		torch.cuda.is_available() and cfg["cuda"] else torch.device("cpu")
	
	len_seq = cfg["len_seq"]
	corpus = data.Corpus(cfg["data"], len_seq)
	
	batch_size = cfg["batch_size"]
	train_data = batchify(corpus.train, batch_size)
	val_data = batchify(corpus.valid, batch_size)
	test_data = batchify(corpus.test, batch_size)
	
	# Build the model
	vocab_size = len(corpus.dictionary)
	if cfg["model_type"] == 'Transformer':
		model = model.TransformerModel(
			vocab_size, cfg["d_emb"], cfg["n_head"], cfg["d_ffn"], cfg["n_layer"], cfg["dropout"]
		).to(device)
	
	criterion = nn.NLLLoss()
	lr = cfg["lr"]
	best_val_loss = None
	
	# At any point you can hit Ctrl + C to break out of training early.
	try:
		for epoch in range(1, cfg["epoch"] + 1):
			epoch_start_time = time.time()
			with autocast("cuda"):
				train()
			val_loss = evaluate(val_data)
			print('-' * 89)
			print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
				  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
											 val_loss, math.exp(val_loss)))
			print('-' * 89)
			# Save the model if the validation loss is the best we've seen so far.
			if not best_val_loss or val_loss < best_val_loss:
				with open(cfg["save_path"], 'wb') as f:
					torch.save(model, f)
				best_val_loss = val_loss
			elif lr >= 0.05:
				# Anneal the learning rate if no improvement has been seen in the validation dataset.
				lr /= cfg["decrease_rate"]
				print("WARNING: for no improvement,lr was decreased from {:5.3f} to {:5.3f}".format(
					lr * cfg["decrease_rate"], lr
				))
	except KeyboardInterrupt:
		print('-' * 89)
		print('Exiting from training early')
	
	# Load the best saved model.
	with open(cfg["save_path"], 'rb') as f:
		model = torch.load(f)
	
	# Run on test data.
	test_loss = evaluate(test_data)
	print('=' * 89)
	print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
		test_loss, math.exp(test_loss)))
	print('=' * 89)
	
	if len(cfg["onnx_path"]) > 3:
		# Export the model in ONNX format.
		export_onnx(cfg["onnx_path"], model_batch_size=1, seq_len=len_seq)
