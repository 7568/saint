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
import torch.nn as nn
import torch
from einops import rearrange


class LSTM(nn.Module):
    """
    input_size - will be 1 in this example since we have only 1 predictor (a sequence of previous values)
    hidden_size - Can be chosen to dictate how much hidden "long term memory" the network will have
    output_size - This will be equal to the prediciton_periods input to get_x_y_pairs
    """

    def __init__(self, input_size, hidden_size, num_layers,output_size, dropout,device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, dropout=dropout)

        self.linear = nn.Linear(hidden_size, output_size)

        self.num_layers = num_layers
        self.output_size = output_size
        self.device = device

    def forward(self, x, hidden=None):
        x = rearrange(x, 'b s 1 n -> s b n')
        # print(x.shape)
        s, b, n = x.shape
        if hidden == None:
            self.hidden = (torch.zeros(self.num_layers, b, self.hidden_size).to(self.device),
                           torch.zeros(self.num_layers, b, self.hidden_size).to(self.device))
        else:
            self.hidden = hidden

        """
        inputs need to be in the right shape as defined in documentation
        - https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        lstm_out - will contain the hidden states from all times in the sequence
        self.hidden - will contain the current hidden state and cell state
        """
        # print(x.shape)
        # print(x.view(len(x), 1, -1).shape)
        # lstm_out, self.hidden = self.lstm(x.view(len(x), 1, -1),
        #                                   self.hidden)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        # print(lstm_out.shape)

        predictions = self.linear(lstm_out[-1])

        return predictions
