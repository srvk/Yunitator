USAGE = """  
python 1-confidence-vm2.py <INPUT FOLDER>
  where <INPUT FOLDER> contains 1 or more .htk features files
"""

RNNPATH='OpenSAT/SSSF/code/predict/RNN'
TOOLSPATH="G/coconut"
NNET='SSSF/code/predict/model/noiseme.old/net.pkl.gz'
PCAMATRIX='SSSF/code/predict/model/noiseme.old/pca.pkl'
SCALINGFACTORS='SSSF/code/predict/model/noiseme.old/scale.pkl'

import sys
if len(sys.argv) < 2:
    print USAGE
    sys.exit(1)

import os, os.path
import numpy
import cPickle
sys.path.append(os.path.expanduser(RNNPATH))
from RNN import RNN
sys.path.append(os.path.expanduser(TOOLSPATH))
from fileutils import smart_open
from fileutils.htk import *
from scipy.io import savemat

# Paths  
INPUTFILE = sys.argv[1] # /vagrant/noisemes/
#PART = "evl.med"
#BASENAME = sys.argv[2]
INPUT_DIR = INPUTFILE+"/feature"
OUTPUT_DIR = INPUTFILE+"/hyp"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load neural network
print "Loading neural network..."
net = RNN(filename = os.path.expanduser(NNET))

# Load PCA matrix and scaling factors
with open(os.path.expanduser(PCAMATRIX), 'rb') as f:
    locals().update(cPickle.load(f))
    with open(os.path.expanduser(SCALINGFACTORS), 'rb') as f:
        w, b = cPickle.load(f)
        pca = lambda x: ((x[:,mask] - mu) / sigma).dot(V) * w + b

# Predict for each recording 
for filename in os.listdir(INPUT_DIR):
    conf = {}
    print "Filename {}".format(filename)
    id,ext = os.path.splitext(filename)
    if ext != '.htk': continue
    print "Predicting for {} ...".format(id)
    feature = pca(readHtk(os.path.join(INPUT_DIR, filename))).astype('float32')
    x = feature.reshape((1,) + feature.shape)
    m = numpy.ones(x.shape[:-1], dtype = 'int32')
    conf[id] = net.predict(x, m)[0]

    # Save predictions   
    with smart_open(os.path.join(OUTPUT_DIR, id + '.confidence.pkl.gz'), 'wb') as f:
        cPickle.dump(conf, f)
        savemat(os.path.join(OUTPUT_DIR, id + '.confidence.mat'), conf)



