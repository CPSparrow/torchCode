import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 2
src_len = torch.Tensor([2, 4]).to(torch.int32)
tgt_len = torch.Tensor([2, 4]).to(torch.int32)
test = "123"
print(repr(src_len))
