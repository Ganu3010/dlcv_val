import torch
import numpy as np

a = torch.tensor(np.load('dataset/images.npy'))

print(a.shape, '\n')

# print(min(a))
# print(max(a))