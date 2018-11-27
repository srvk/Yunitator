import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import numpy

class Net(nn.Module):
    def __init__(self, nInput, nHidden, nOutput):
        super(Net, self).__init__()
        self.gru = nn.GRU(nInput, nHidden, 1, bidirectional = True)
        self.fc = nn.Linear(nHidden * 2, nOutput)
        # Xavier Glorot initialization
        nn.init.orthogonal_(self.gru.weight_ih_l0); nn.init.constant_(self.gru.bias_ih_l0, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0); nn.init.constant_(self.gru.bias_hh_l0, 0)
        nn.init.orthogonal_(self.gru.weight_ih_l0_reverse); nn.init.constant_(self.gru.bias_ih_l0_reverse, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0_reverse); nn.init.constant_(self.gru.bias_hh_l0_reverse, 0)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        # Returns log probabilities
        # Both input and output are PackedSequences
        x = self.gru(x)[0]
        return PackedSequence(F.softmax(self.fc(x[0]), dim = -1), x[1])
