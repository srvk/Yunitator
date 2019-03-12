import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import numpy

class Net(nn.Module):
    def __init__(self, nInput, nHidden, nLayers, nOutput):
        super(Net, self).__init__()

        self.gru = nn.GRU(nInput, nHidden, nLayers, bidirectional = True)
        self.fc = nn.Linear(nHidden * 2, nOutput) # Bidirectional, so the size of the output is 2*nHidden
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
        x = PackedSequence(F.softmax(self.fc(x[0]), dim=-1), x[1])
        return x

    def predict(self, x, batch_size = 30):
        # Predict in batches.
        # x is a list of feature matrices, possibly of different lengths.
        # Returns a list of prediction matrices.
        ind = numpy.argsort([len(z) for z in x])[::-1]    # Sort the matrices in x by length in descending order
        y = [None] * len(x)
        for start in range(0, len(x), batch_size):
            end = min(len(x), start + batch_size)
            lengths = [len(x[i]) for i in ind[start:end]]
            input = numpy.zeros((end - start, lengths[0]) + x[start].shape[1:], dtype = 'float32')
            for i in range(start, end):
                input[i - start, :len(x[ind[i]])] = x[ind[i]]
            input = Variable(torch.from_numpy(input), requires_grad = False).cuda()
            input = pack_padded_sequence(input, lengths, batch_first = True)
            output = self.forward(input)
            output = pad_packed_sequence(output, batch_first = True)[0].data.cpu().numpy()
            for i in range(start, end):
                y[ind[i]] = output[i - start, :len(x[ind[i]])]
        return y
