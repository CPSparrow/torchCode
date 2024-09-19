import torch
import torch.nn as nn

drop = nn.Dropout(p=0.1)
score = torch.Tensor([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
])
out = drop(score)
print(score, '\n', out)
