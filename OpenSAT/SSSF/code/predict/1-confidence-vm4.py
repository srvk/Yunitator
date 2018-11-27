USAGE = """  
python 1-confidence-vm2.py <INPUT FOLDER>
  where <INPUT FOLDER> contains 1 or more .htk features files
"""

#RNNPATH='~/OpenSAT/SSSF/code/predict/RNN'
#TOOLSPATH="~/G/coconut"
#NNET='~/OpenSAT/SSSF/code/predict/model/noiseme.old/net.pkl.gz'
#PCAMATRIX='~/OpenSAT/SSSF/code/predict/model/noiseme.old/pca.pkl'
#SCALINGFACTORS='~/OpenSAT/SSSF/code/predict/model/noiseme.old/scale.pkl'
RNNPATH='SSSF/code/predict/RNN'
TOOLSPATH="G/coconut"
NNET='SSSF/code/predict/model/noiseme.old/net.pkl.gz'
PCAMATRIX='SSSF/code/predict/model/noiseme.old/pca.pkl'
SCALINGFACTORS='SSSF/code/predict/model/noiseme.old/scale.pkl'

import sys
import ipdb
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
classfile = open("noisemeclasses_sum.txt", 'rb')
for line in classfile:
	classes.append(line.rstrip('\n'))


# Paths  
INPUTFILE = sys.argv[1] # /vagrant/noisemes/
INPUT_DIR = INPUTFILE+"/feature"
OUTPUT_DIR = INPUTFILE+"/hyp_sum"
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

    result_ = conf[id]

    # Add classes 1 and 2 (speech english and speech non english)
    # to create a class " Speech "
    result = numpy.zeros((result_.shape[0], result_.shape[1] - 1))
    result[:, 0] = result_[:, 0]
    result[:, 1] = result_[:, 1] + result_[:, 2]
    result[:, 2:] = result_[:, 3:]

    # Output RTTM
    most_likely = result.argmax(axis=1)
    confidence = result.max(axis=1)

    length_sample = len(most_likely)
    time_frame = numpy.arange(0, length_sample) * 0.1

    with open(os.path.join(OUTPUT_DIR, id + ".rttm"), "w") as rttm:

        t_start = time_frame[0]

        for t in range(length_sample):
            # If current frame is from a different class then previous frame,
            # then write class that ended at last frame (t-1 !).
            # Also avoid problem: for t=0, most_likely[t-1] = most_likely[-1]
            # which is the last item of the vector !
            if t == 0:
                continue
            if most_likely[t] != most_likely[t-1] :
                if classes[most_likely[t-1]] == "speech":
                    # If class is speech, write as SPEAKER
                    # get duration of class
                    time_elapse = time_frame[t] - t_start
                    # get class that just ended
                    frame_class = most_likely[t-1]
                    # get confidence of system
                    confidence = result[t][frame_class]
                    rttm.write(u"SPEAKER\t{}\t{}\t{}\t{}\t<NA>\t<NA>\t{}\t{}\t<NA>\n".format
		        (id, 1, t_start, time_elapse, classes[most_likely[t-1]],
                         confidence ))
                    t_start = time_frame[t]
                else:
                    # If class is noise, write as NON-SPEECH
                    # get duration of class
                    time_elapse = time_frame[t] - t_start
                    # get class that just ended
                    frame_class = most_likely[t-1]
                    # get confidence of system
                    confidence = result[t][frame_class]
                    rttm.write(u"NON-SPEECH\t{}\t{}\t{}\t{}\t<NA>\t<NA>\t{}\t{}\t<NA>\n".format
		        (id, 1, t_start, time_elapse, classes[most_likely[t-1]],
                         confidence ))
                    t_start = time_frame[t]
