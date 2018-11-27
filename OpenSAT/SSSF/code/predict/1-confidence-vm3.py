USAGE = """  
python 1-confidence-vm2.py <INPUT FOLDER>
  where <INPUT FOLDER> contains 1 or more .htk features files
"""

RNNPATH='SSSF/code/predict/RNN'
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

# classes.txt contains class name strings, one per line
classes = []
classfile = open("noisemeclasses.txt", 'rb')
for line in classfile:
	classes.append(line.rstrip('\n'))


# Paths  
INPUTFILE = sys.argv[1] # /vagrant/noisemes/
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

    result = conf[id]
    # Output RTTM
    most_likely = result.argmax(axis=1)
    confidence = result.max(axis=1)

    length_sample = len(most_likely)
    time_frame = numpy.arange(0, length_sample) * 0.1

    with open(os.path.join(OUTPUT_DIR, id + ".rttm"), "w") as rttm:

        t_start = time_frame[0]

        for t in range(length_sample):
            if most_likely[t] != most_likely[t-1]:
                time_elapse = time_frame[t] - t_start
                frame_class = most_likely[t]
                confidence = result[t][frame_class]
                rttm.write(u"SPEAKER\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format
		    (id, 1, t_start, time_elapse, classes[most_likely[t]], "<NA>", "<NA>", confidence ))
                t_start = time_frame[t]


