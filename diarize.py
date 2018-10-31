import sys, os, os.path
sys.path.append(os.path.expanduser("~/G/coconut"))
from fileutils.htk import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Net import Net
import cPickle
import numpy

# Load model
net = Net(50, 200, 4) #.cuda()
net.load_state_dict(torch.load('model.pt', map_location = lambda storage, loc: storage))

INPUT_FILE = sys.argv[1]        # Feature file containing 6,669-dim HTK-format features
OUTPUT_FILE = sys.argv[2]       # RTTM file to write the results to

# Load PCA matrices
with open('pca-self.pkl', 'rb') as f:
    data = cPickle.load(f)
mask, mu, sigma, V, w, b = data['mask'], data['mu'], data['sigma'], data['V'], data['w'], data['b']
pca = lambda feat: ((feat[:, mask] - mu) / sigma).dot(V) * w + b

# Load input feature and predict
feat = pca(readHtk(INPUT_FILE))
input = Variable(torch.from_numpy(numpy.expand_dims(feat, 0).astype('float32'))) #.cuda()
input = pack_padded_sequence(input, [len(feat)], batch_first = True)
output = net(input).data.data.cpu().numpy()

# Print the predictions in RTTM format
class_names = ['SIL', 'CHI', 'MAL', 'FEM']
nClasses = len(class_names)
FRAME_LEN = 0.1

output = output == output.max(axis = 1, keepdims = True)
z = numpy.zeros((nClasses, 1), dtype = 'bool')
output = numpy.hstack([z, output.T, z])
cls_ids, starts = (~output[:, :-1] & output[:, 1:]).nonzero()
_, ends = (output[:, :-1] & ~output[:, 1:]).nonzero()

key = os.path.splitext(os.path.basename(OUTPUT_FILE))[0]
with open(OUTPUT_FILE, 'w') as f:
    for cls, start, end in zip(cls_ids, starts, ends):
        f.write('SPEAKER %s 1 %.1f %.1f <NA> <NA> %s <NA> <NA>\n' % \
            (key, start * FRAME_LEN, (end - start) * FRAME_LEN, class_names[cls]))
