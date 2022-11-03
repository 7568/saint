# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/11/2
Description:
"""
import torch
from torch import nn
from torch.autograd.grad_mode import F

input = torch.randn(3, 5, requires_grad=True)
target = torch.randint(5, (3,), dtype=torch.int64)
loss =nn.CrossEntropyLoss()(input, target)
print(loss.item())
loss.backward()