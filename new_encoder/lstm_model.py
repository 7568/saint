# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/11/3
Description:
"""
# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/11/3
Description:
"""
import os
import sys

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
from einops import rearrange
import torch.nn as nn
from torch import Tensor


class LSTM(nn.Module):
    def __init__(
            self,
            input_feature_size,
            hidden_size,
            num_layers,
            dropout=0.,
            num_classes=2
    ) -> None:
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_feature_size, hidden_size, num_layers, dropout=dropout)

        self.fc = nn.Linear(512, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # print('_forward_impl',x.shape)
        x = rearrange(x, 'b s 1 n -> s b n')
        x, (hn, cn)= self.rnn(x)
        # print(x)
        # print(x.shape)
        x = x[-1, :, :]
        # print(x.shape)
        # x = rearrange(x, 'b n -> b 1 n')
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)