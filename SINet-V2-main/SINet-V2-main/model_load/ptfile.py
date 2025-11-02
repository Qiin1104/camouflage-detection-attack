import copy

import numpy as np
import torch

path = '../pre_model/res2net50_v1b_26w_4s-3cf99910.pth'
path1 = '../pre_model/sinetv2-666.pth'
m1 = torch.load(path)
m2 = torch.load(path1)
i = 0
for k, v in m1.items():
    print(k)
    i += 1
print(i)
i = 0
for k, v in m2.items():
    print(k)
    i += 1
print(i)

