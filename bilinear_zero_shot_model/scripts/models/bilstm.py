import torch
import torch.nn as nn

# rnnlib package for Bidirectional Layer Norm LSTM
# https://github.com/daehwannam/pytorch-rnn-library
from scripts.models.rnnlib.seq import LayerNormLSTM

class BiLstm(nn.Module):
    def __init__(self, vocabnum, seq_lens):
        super(BiLstm, self).__init__()

        self.vocabnum = vocabnum
        self.seq_lens = seq_lens
        self.num_directions = 2 # Bidirectional

        # Attention
        self.attention = Attention(20, 512 * 2)

        self.batchnorm1 = nn.BatchNorm1d(self.vocabnum)
        self.dropout_layer = nn.Dropout(p=0.5)##nn.Dropout1d(p=0.5)
        
        self.bi_lstm = LayerNormLSTM(self.vocabnum, 512, 2, dropout=0, r_dropout=0,
                             bidirectional=True, layer_norm_enabled=True)

        self.batchnorm2 = nn.BatchNorm1d(512 * 2)

    def forward(self,x):
        # batch_embedded (n, l, c) -> (n, c, l)
        x = x.permute(0,2,1)
        x = self.batchnorm1(x)
        x = self.dropout_layer(x)
        x = x.permute(2,0,1) # (n, c, l) -> (l, n, c)
        # 13,64,100
        x, _ = self.bi_lstm(x, None)
        # 13,64,1024
        x = x.permute(1,2,0) # (l, n, c) -> (n, c, l)
        x = self.batchnorm2(x)
        x = x.permute(0,2,1) # (n, c, l) -> (n, l, c)
        x, _ = self.attention(x)
        x = self.dropout_layer(x.unsqueeze(2)).squeeze(2) # (n, c, l)
        return x


class Attention(nn.Module):
    def __init__(self, attention_size, hidden_size):
        super(Attention, self).__init__()
        self.attention_size = attention_size
        self.hidden_size = hidden_size
        # Trainable parameters
        self.w_omega = nn.Parameter(torch.randn(self.hidden_size, self.attention_size) * 0.05)
        self.b_omega = nn.Parameter(torch.randn(self.attention_size) * 0.05)
        self.u_omega = nn.Parameter(torch.randn(self.attention_size) * 0.05)

    def forward(self, inputs):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        # the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = torch.tanh(torch.matmul(inputs, self.w_omega) + self.b_omega)
        v = v.view(-1, v.size(1), self.attention_size)

        vu = torch.matmul(v, self.u_omega)
        alphas = torch.softmax(vu, dim=1)
        output = torch.sum(inputs * alphas.unsqueeze(-1), dim=1)
        return output, alphas