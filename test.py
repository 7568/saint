# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/25
Description:
"""
import numpy as np

import torch

a = torch.randn((2,3,1))
print(a.shape)
torch.squeeze(a,2)
print(a.shape)