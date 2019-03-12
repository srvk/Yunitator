import numpy
import struct
from fileutils import smart_open

def readPfile(filename):
    """
    Reads the contents of a pfile. Returns a tuple (features, labels), where
    both elements are lists of 2-D numpy arrays. Each element of a list
    corresponds to a sentence; each row of a 2-D array corresponds to a frame.
    In the case where the pfile doesn't contain labels, "labels" will be None.
    """

    with smart_open(filename, "rb") as f:
        # Read header
        # Assuming all data are consistent
        for line in f:
            tokens = line.decode().split()
            if tokens[0] == "-pfile_header":
                headerSize = int(tokens[4])
            elif tokens[0] == "-num_sentences":
                nSentences = int(tokens[1])
            elif tokens[0] == "-num_frames":
                nFrames = int(tokens[1])
            elif tokens[0] == "-first_feature_column":
                cFeature = int(tokens[1])
            elif tokens[0] == "-num_features":
                nFeatures = int(tokens[1])
            elif tokens[0] == "-first_label_column":
                cLabel = int(tokens[1])
            elif tokens[0] == "-num_labels":
                nLabels = int(tokens[1])
            elif tokens[0] == "-format":
                format = tokens[1].replace("d", "i")
            elif tokens[0] == "-end":
                break
        nCols = len(format)
        dataSize = nFrames * nCols

        # Read sentence index
        f.seek(headerSize + dataSize * 4)
        index = struct.unpack(">%di" % (nSentences + 1), f.read(4 * (nSentences + 1)))

        # Read data
        f.seek(headerSize)
        features = []
        labels = []
        sen = 0
        for i in range(nFrames):
            if i == index[sen]:
                features.append([])
                labels.append([])
                sen += 1
            data = struct.unpack(">" + format, f.read(4 * nCols))
            features[-1].append(data[cFeature : cFeature + nFeatures])
            labels[-1].append(data[cLabel : cLabel + nLabels])
        features = [numpy.array(x) for x in features]
        labels = [numpy.array(x) for x in labels] if nLabels > 0 else None

    return (features, labels)

def writeBytes(f, string):
	f.write(string.encode())

def writePfile(filename, features, labels = None):
    """
    Writes "features" and "labels" to a pfile. Both "features" and "labels"
    should be lists of 2-D numpy arrays. Each element of a list corresponds
    to a sentence; each row of a 2-D array corresponds to a frame. In the case
    where there is only one label per frame, the elements of the "labels" list
    can be 1-D arrays.
    """

    nSentences = len(features)
    nFrames = sum(len(x) for x in features)
    nFeatures = len(numpy.array(features[0][0]).ravel())
    nLabels = len(numpy.array(labels[0][0]).ravel()) if labels is not None else 0
    nCols = 2 + nFeatures + nLabels
    headerSize = 32768
    dataSize = nFrames * nCols

    with smart_open(filename, "wb") as f:
        # Write header
        writeBytes(f, "-pfile_header version 0 size %d\n" % headerSize)
        writeBytes(f, "-num_sentences %d\n" % nSentences)
        writeBytes(f, "-num_frames %d\n" % nFrames)
        writeBytes(f, "-first_feature_column 2\n")
        writeBytes(f, "-num_features %d\n" % nFeatures)
        writeBytes(f, "-first_label_column %d\n" % (2 + nFeatures))
        writeBytes(f, "-num_labels %d\n" % nLabels)
        writeBytes(f, "-format dd" + "f" * nFeatures + "d" * nLabels + "\n")
        writeBytes(f, "-data size %d offset 0 ndim 2 nrow %d ncol %d\n" % (dataSize, nFrames, nCols))
        writeBytes(f, "-sent_table_data size %d offset %d ndim 1\n" % (nSentences + 1, dataSize))
        writeBytes(f, "-end\n")

        # Write data
        f.seek(headerSize)
        for i in range(nSentences):
            for j in range(len(features[i])):
                f.write(struct.pack(">2i", i, j))
                f.write(struct.pack(">%df" % nFeatures, *numpy.array(features[i][j]).ravel()))
                if labels is not None:
                    f.write(struct.pack(">%di" % nLabels, *numpy.array(labels[i][j].astype(int)).ravel()))

        # Write sentence index
        index = numpy.cumsum([0] + [len(x) for x in features])
        f.write(struct.pack(">%di" % (nSentences + 1), *index))
