import numpy
import struct
from fileutils import smart_open

def readHtk(filename, chunk_size=None, preSamples=None):
    """
    Reads the features in a HTK file, and returns them in a 2-D numpy array.
    chunk_size: integer specifying number of samples per chunk.
    preSamples: integer specifying the number of samples to prepend to
                a chunk to try and deal with issues at chunk boundaries.
                Safe to assume that if chunk_size is not None, preSamples
                will also not be None.
    """
    # Only do chunking if chunk_size is passed to the function.
    if chunk_size is not None:
        assert chunk_size > 0, "chunk_size needs to be > 0"
        with smart_open(filename, "rb") as f:
            nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
            assert nSamples > 0, "nSamples needs to be > 0"

            # If the size of the features is less than the chunk size.
            if nSamples < chunk_size:
                chunk_size = nSamples

            # Iterate over all full chunks first.
            for i in range(nSamples // chunk_size):
                # We want to add a few samples to the beginning of each chunk,
                # but only after the first one.
                if i == 0:
                    readSize = chunk_size * sampSize
                    dataSize = readSize / 4
                    outputSize = chunk_size
                else:
                    readSize = (chunk_size + preSamples) * sampSize
                    dataSize = readSize / 4
                    outputSize = chunk_size + preSamples

                data = struct.unpack(">%df" % (dataSize), f.read(readSize))
                yield numpy.array(data).reshape(outputSize, sampSize // 4)

                # Move the file cursor back so that the next chunk reuses
                # some of the same samples.
                f.seek(-(preSamples * sampSize), 1)

            # Whatever remains after the last full chunk size.
            chunk_size = nSamples - (chunk_size * (nSamples // chunk_size)) + preSamples
            if chunk_size > 0:
                data = struct.unpack(">%df" % (chunk_size * sampSize / 4), f.read(chunk_size * sampSize))
                yield numpy.array(data).reshape(chunk_size, sampSize // 4)
    else:
        with smart_open(filename, "rb") as f:
            #Read header
            nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">iihh", f.read(12))
            # Read data
            data = struct.unpack(">%df" % (nSamples * sampSize / 4), f.read(nSamples * sampSize))
            # print(type(data), len(data), nSamples, sampSize, nSamples*sampSize/4)
            yield numpy.array(data).reshape(nSamples, sampSize // 4)


def writeHtk(filename, feature, sampPeriod, parmKind):
    """
    Writes the features in a 2-D numpy array into a HTK file.
    """
    with smart_open(filename, "wb") as f:
        # Write header
        nSamples = feature.shape[0]
        sampSize = feature.shape[1] * 4
        f.write(struct.pack(">iihh", nSamples, sampPeriod, sampSize, parmKind))

        # Write data
        f.write(struct.pack(">%df" % (nSamples * sampSize / 4), *feature.ravel()))
