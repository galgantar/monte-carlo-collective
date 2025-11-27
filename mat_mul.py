import torch
import numpy as np

if torch.cuda.is_available():
  device = 'cuda'
  print('gpu ok')
else:
  device = 'cpu'
a = torch.randn(22,22,22)
b = torch.randn(22,22,22)
a = a.cuda()
b = b.cuda()
while True:
  torch.bmm(a,b)