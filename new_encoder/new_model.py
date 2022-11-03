# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/28
Description:
"""
import math

import torch
import torch.nn as nn
from einops import rearrange


class simple_MLP(nn.Module):
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )

    def forward(self, x):
        return self.layers(x)


class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        # self.encoder = nn.Linear(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)

        # self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        # nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        # nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input):
        # enc = self.drop(self.encoder(input))
        output, hidden = self.rnn(input)
        output = self.drop(output)
        return output

    def init_hidden(self, bsz=5):
        # params = self.parameters()
        # params.dtype = torch.float32
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        # self.decoder = nn.Linear(ninp, ntoken)

        # self.init_weights()

    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        # nn.init.zeros_(self.decoder.bias)
        # nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src):

        # src = self.encoder(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # output = self.decoder(output)
        # return F.log_softmax(output, dim=-1)
        return output


class NewModel(nn.Module):
    def __init__(self, input_feature_size,expand_size=1):
        super(NewModel, self).__init__()


        hidden_size = input_feature_size * 2*expand_size  # 在hidden中的feature大小
        num_layers = 6
        dropout = 0.1
        self.lstm = LSTMModel(input_feature_size, hidden_size, num_layers*expand_size, dropout=dropout)
        atten_block_n = 6*expand_size
        nhead = 2
        nhid = 512
        nlayers = 1
        self.axial_attention = nn.ModuleList([TransformerModel(50, nhead, nhid, nlayers, dropout=dropout),
                                              TransformerModel(160+hidden_size, nhead * 2, nhid*2 * expand_size, nlayers,
                                                               dropout=dropout)] * atten_block_n)
        self.simple_MLP = simple_MLP([160+hidden_size, 1024, 2])

    def forward(self, x, x_flatten):
        # print(x.device)
        x = self.lstm(x)
        # print(x.shape)
        x = x[:, -1, :]
        # print(x.shape)
        # print(x_flatten.shape)
        x = torch.cat((x, x_flatten), 1)
        # print(x.shape)

        # print(x.device)
        x = rearrange(x, 'b n -> 1 n b')
        # print(x.device)
        _, d, _ = x.shape
        for i, attention in enumerate(self.axial_attention):
            # print(attention)
            # print(x.shape)
            if i % 2 == 0:
                # print(x.device)
                x = attention(x)
            else:
                x = rearrange(x, '1 n b -> 1 b n')
                x = attention(x)
                x = rearrange(x, '1 b n -> 1 n b')
        x = rearrange(x, '1 n b -> b n')
        # print(x.shape)
        # print(x.shape)
        # print(self.simple_MLP)
        out = self.simple_MLP(x)
        return x, out
