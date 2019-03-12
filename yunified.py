# Unified scipt for Noisemes, Yunitator, and TALNet
#
#
# Usage: yunified.py SCRIPT INPUT_DIR YUNITATOR_OUTPUT_FILE HTK_CHUNKSIZE
#
# SCRIPT: which script to run [yunitator, noisemes]
# INPUT_DIR: the input directory (eg. /vagrant/data)
# HTK_CHUNKSIZE: The number of frames to use for each chunk (10 frames per second)

# ---------------------------------------------------------------------
# -------------------------- PATH VARIABLES ---------------------------
# ---------------------------------------------------------------------


RNNPATH='OpenSAT/SSSF/code/predict/RNN'
TOOLSPATH="G/coconut"
SCALINGFACTORS='OpenSAT/SSSF/code/predict/model/noiseme.old/scale.pkl'
NOISEMES_CLASSES='OpenSAT/noisemeclasses_sum.txt'


# ---------------------------------------------------------------------
# ------------------------------ PACKAGES -----------------------------
# ---------------------------------------------------------------------


import os
import sys
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import _pickle as cPickle
import pickle
import numpy
from scipy.io import savemat

sys.path.append(os.path.expanduser(TOOLSPATH))
from fileutils import smart_open
from fileutils.htk import readHtk

sys.path.append(os.path.expanduser(RNNPATH))
from RNN import RNN
from Yunitator.Net import Net

import warnings                     # Not safe at all, but it's to discard
warnings.filterwarnings('ignore')   # ComplexWarning: Casting complex values to real discards the imaginary part
                                    # that occurs when applying the LDA (some coefficients of the matrix can have
                                    # an imaginary part)
# ---------------------------------------------------------------------
# --------------------------- BEGIN SCRIPT ----------------------------
# ---------------------------------------------------------------------


# Script options
YUNITATOR = 'yunitator'
NOISEMES = 'noisemes'
ENGLISH = 'english'
UNIVERSAL = 'universal'
OLD = 'old'

# Args
try:
    SCRIPT = sys.argv[1]  # Which script to run [yunitator, noisemes]
    INPUT_DIR = sys.argv[2].rstrip('/')  # HTK Dir (eg. /vagrant/data/)
    HTK_CHUNKSIZE = int(sys.argv[3])
    try:
        MODE = sys.argv[4]  # Which mode tu use [english, universal, old], only used if SCRIPT == YUNITATOR
    except IndexError:
        MODE = OLD
except IndexError:
    print("WRONG NUMBER INPUTS")
    exit()

# Choose Matrix variables
if SCRIPT == YUNITATOR:
    if MODE == OLD:
        TRANSFMATRIX = 'Yunitator/reductions/pca-old.pkl'         # PCA of size 50
        NNET = 'Yunitator/models/model-old.pt'
    elif MODE == ENGLISH:
        TRANSFMATRIX = 'Yunitator/reductions/pca-english.pkl'      # PCA of size 150
        NNET = 'Yunitator/models/model-english.pt'
    elif MODE == UNIVERSAL:
        TRANSFMATRIX = 'Yunitator/reductions/lda-universal.pkl'    # LDA of size 150
        NNET = 'Yunitator/models/model-universal.pt'
        print("GOT IT !")
elif SCRIPT == NOISEMES:
    TRANSFMATRIX = 'OpenSAT/SSSF/code/predict/model/noiseme.old/pca.pkl'
    NNET = 'OpenSAT/SSSF/code/predict/model/noiseme.old/net.pkl.gz'

# Prepare output directories
if SCRIPT == YUNITATOR:
    INPUT_DIR += "/Yunitemp"
    OUTPUT_DIR = INPUT_DIR
elif SCRIPT == NOISEMES:
    OUTPUT_DIR = INPUT_DIR + "/hyp_sum"
    INPUT_DIR += "/feature"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


# Load neural network
if SCRIPT == YUNITATOR:
    if MODE == OLD:
        net = Net(50, 200,1, 4) #.cuda()
    elif MODE == UNIVERSAL:
        net = Net(150, 300, 2, 4)
    elif MODE == ENGLISH:
        net = Net(150, 200, 1, 4)
    net.load_state_dict(torch.load(NNET, map_location = lambda storage, loc: storage))
elif SCRIPT == NOISEMES:
    net = RNN(filename=os.path.expanduser(NNET))


# Get class names
# noisemeclasses_sum.txt contains class name strings, one per line
if SCRIPT == YUNITATOR:
    class_names =['SIL', 'CHI', 'MAL', 'FEM']
elif SCRIPT == NOISEMES:
    class_names = []    
    classfile = open(NOISEMES_CLASSES, 'rb')
    for line in classfile:
        class_names.append(line.decode('utf-8').rstrip('\n'))


# Load PCA matrix and scaling factors
if SCRIPT == YUNITATOR:
    with open(TRANSFMATRIX, 'rb') as f:
        data = pickle.load(f, encoding="ASCII")
        if MODE == OLD or MODE == ENGLISH:
            mask, mu, sigma, V, w, b = data['mask'], data['mu'], data['sigma'], data['V'], data['w'], data['b']
            reductor = lambda feat: ((feat[:, mask] - mu) / sigma).dot(V) * w + b
        elif MODE == UNIVERSAL:
            mask, V, mu, sigma = data['mask'], data['V'], data['mu'], data['sigma']
            print(mask.shape)
            print(V.shape)
            print(mu.shape)
            print(sigma.shape)
            print(sys.version)
            reductor = lambda feat: (feat.dot(V)[:,mask] - mu) / sigma # Before :(feat.dot(V)[:, mask] - mu) / sigma
            print("GOT IT !2")
elif SCRIPT == NOISEMES:
    with open(os.path.expanduser(TRANSFMATRIX), 'rb') as f:
        locals().update(cPickle.load(f, encoding="latin1"))
        with open(os.path.expanduser(SCALINGFACTORS), 'rb') as f:
            w, b = cPickle.load(f, encoding="latin1")
            reductor = lambda feat: ((feat[:, mask] - mu) / sigma).dot(V) * w + b

# These are chunking parameters.
# ex: HTK_chunksize = 2000
if SCRIPT == YUNITATOR:
    preSamples = 0
elif SCRIPT == NOISEMES:
    preSamples = 30 

for file in os.listdir(INPUT_DIR):
    # Load input feature and predict
    filename, extension = os.path.splitext(os.path.split(file)[1])
    conf = {}

    # noisemes needs a variable to remember the last timestep when chunking
    last_t = 0  
    chunks = 0
    feat = readHtk(INPUT_DIR+"/"+file)

    if SCRIPT == YUNITATOR:
        feature = reductor(feat)
        input = Variable(torch.from_numpy(numpy.expand_dims(feature, 0).astype('float32')))
        input = pack_padded_sequence(input, [len(feature)], batch_first = True)
        output = net(input).data.data.cpu().numpy()

        output = output == output.max(axis = 1, keepdims = True)
        z = numpy.zeros((len(class_names), 1), dtype = 'bool')
        output = numpy.hstack([z, output.T, z])
        cls_ids, starts = (~output[:, :-1] & output[:, 1:]).nonzero()
        _, ends = (output[:, :-1] & ~output[:, 1:]).nonzero()

        with open(OUTPUT_DIR+"/"+filename+".rttm.sorted", 'a') as f:
            for cls, start, end in zip(cls_ids, starts, ends):
                print("%s\t%s\t%s" % (cls,start,end))
                f.write("SPEAKER {} 1 {:.1f} {:.1f} <NA> <NA> {} <NA> <NA>\n".format(
                    filename+".rttm", (start+(chunks*HTK_CHUNKSIZE)) * 0.1, (end - start) * 0.1, class_names[cls]))
            chunks += 1
    elif SCRIPT == NOISEMES:
        feature = reductor(feat).astype('float32')
        x = feature.reshape((1,) + feature.shape)
        m = numpy.ones(x.shape[:-1], dtype='int32')
        conf[filename] = net.predict(x, m)[0]

        # Save predictions
        with smart_open(os.path.join(OUTPUT_DIR, filename + '.confidence.pkl.gz'), 'wb') as f:
            cPickle.dump(conf, f)
            savemat(os.path.join(OUTPUT_DIR, filename + '.confidence.mat'), conf)
        result_ = conf[filename]

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

        with open(os.path.join(OUTPUT_DIR, filename + ".lab"), "a") as lab:
            t_start = time_frame[0]+last_t
            new_t = 0

            for t in range(length_sample-preSamples):
                # Add the last timestep if this is the first of the chunk.
                time_frame[t] += last_t

                # If current frame is from a different class then previous frame,
                # then write class that ended at last frame (t-1 !).
                # Also avoid problem: for t=0, most_likely[t-1] = most_likely[-1]
                # which is the last item of the vector !
                if t == 0:
                    continue

                # write class only if it correspond to "speech" !
                if most_likely[t] != most_likely[t-1] :
                    # if class_names[most_likely[t-1]] == "speech":
                    #     # get duration of class
                    #     t_end = time_frame[t]
                    #     lab.write(u"{:.1f} {:.1f} speech\n".format(t_start+(0.1*chunks), t_end+(0.1*chunks)))
                    #     t_start = time_frame[t]
                    # else:
                    #     t_start = time_frame[t]

                    # get duration of class
                    t_end = time_frame[t]
                    # Choose rttm type label for speech and everything else. This may need to change to fit convention
                    type_label = "SPEAKER" if class_names[most_likely[t-1]] == 'speech' else "NON-SPEECH"
                    lab.write(u"{:.1f} {:.1f} {} {}\n".format(t_start+(0.1*chunks), t_end+(0.1*chunks), class_names[most_likely[t-1]], type_label))
                    t_start = time_frame[t]

                new_t = time_frame[t]
            last_t = new_t
            chunks += 1
