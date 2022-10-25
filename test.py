# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/25
Description:
"""
import numpy as np
x = np.array([[[0], [1], [2]]])
print(x.shape)

print(np.squeeze(x).shape)

print(np.squeeze(x, axis=(0,)).shape)
