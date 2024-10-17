###############################################################################
# Language Modeling on Wikitext-2
#
# This file generates new sentences sampled from the language model.
#
###############################################################################
import argparse
import json

import torch

import data

with open("./utils/cfg.json") as cfg_file:
	cfg = json.load(cfg_file)
# Set the random seed manually for reproducibility.
torch.manual_seed(cfg["cuda"])
device = torch.device("cuda") if (torch.cuda.is_available() and
								  cfg["cuda"]) else torch.device("cpu")

if cfg["temperature"] < 1e-3:
	parser.error("--temperature has to be greater or equal 1e-3.")

with open(cfg["check_point"], 'rb') as f:
	model = torch.load(f, map_location=device)
model.eval()

corpus = data.Corpus(cfg["data"], cfg["batch_size"])
vocab_size = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
src_seq = torch.randint(vocab_size, (1, 1), dtype=torch.long).to(device)

with open(cfg["tgt_path"], 'w') as tgt_file:
	with torch.no_grad():  # no tracking history
		for i in range(cfg["words"]):
			assert is_transformer_model
			output = model(src_seq, False)
			word_weights = output[-1].squeeze().div(cfg["temperature"]).exp().cpu()
			word_idx = torch.multinomial(word_weights, 1)[0]
			word_tensor = torch.Tensor([[word_idx]]).long().to(device)
			src_seq = torch.cat([src_seq, word_tensor], 0)
			
			word = corpus.dictionary.idx2word[word_idx]
			
			tgt_file.write(word + ('\n' if i % 20 == 19 else ' '))
			
			if i % cfg["generate_log_interval"] == 0:
				print('| Generated {}/{} words'.format(i, cfg["words"]))
