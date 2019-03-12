#! /usr/bin/python 
import os 
import csv
import sys 
import numpy
import gzip
import bz2
import datetime
import resource

from .. import parmap

# ---------------------------------------------------
#   Meaningful Timing
# ---------------------------------------------------

def timing (dt):
        u = resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
        s = resource.getrusage(resource.RUSAGE_SELF).ru_stime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
        t = (datetime.datetime.now()-dt).replace(microsecond=0)
        print("Elapsed Time", t, 'with {:.1f}% user'.format(u/t.total_seconds()*100), 'and {:.1f}% system'.format(s/t.total_seconds()*100))


# ---------------------------------------------------
#   Various Filtering Utilities
# ---------------------------------------------------

def filterFeatures(fileSet,outpath,index):
	#load the indices
	column_indices = open(index).readline().rstrip("\n").split(",")
	print("Reducing to "+str(len(column_indices))+" dimensions")

	#start filtering each of the files in the list
	outputSet = ()
	for filepath in fileSet:
		filepath = filepath.rstrip("\n")
		basename = os.path.basename(filepath)
		output_file = outpath+"/"+basename
		outfilePath = filterFile(filepath,output_file,column_indices)
		outputSet.append(outfilePath)

	return outputSet

def filterFile(filepath,output_file,column_indices,inDelim=';',outDelim=' '):
	filepath = filepath.rstrip("\n")
	csv_file = open(filepath,"r")
	fh = open(output_file, 'w')
	reader = csv.reader(csv_file, delimiter=inDelim)
	wtr = csv.writer(fh,delimiter=outDelim)
	for row in reader:
		wtr.writerow([row[int(i)] for i in column_indices])
	fh.close()
	return output_file

def filterColumns(filepath,column_indices,inDelim=';'):
	filepath = filepath.rstrip("\n")
	raw_features = numpy.loadtxt(filepath,delimiter=inDelim)
	filter_features = raw_features[:,column_indices]
	return filter_features


# ---------------------------------------------------
#   Normalization/ Standardization
# ---------------------------------------------------

def standardize(data,mean,var):
	#subtract mean
	centeredData = data-mean
	#divide by variance
	zscoreData = centeredData/var

	return zscoreData

def standardizeFile(filepath,outfilepath,mean,var):
	#load file 
	filedata = numpy.loadtxt(filepath)
	data = standardize(filedata,mean,var)
	numpy.savetxt(outfilepath,data)
	return outfilepath

def performWhitening(eigVal,eigVec,epsilon=0.0):
	normalizer = 1/(numpy.sqrt(eigVal) + epsilon)
	return eigVec*normalizer

def pcaTransform(data,eigVal,eigVec,n_comps=300):
	eigVal = eigVal[:n_comps]
	eigVec = eigVec[:,:n_comps]
	eigVecWhite = performWhitening(eigVal,eigVec)
	xTransform = numpy.dot(data,eigVecWhite)
	return xTransform


# ---------------------------------------------------
#   File Utilities
# ---------------------------------------------------

def get_files(dir,type):
        filenames = list()
        pathList = os.listdir(dir)
        for path in pathList:
                path = dir + '/' + path
                if os.path.isdir(path):
                        filenames.extend( get_files(path) )
                else:
                        if(os.path.basename(path).endswith("."+type)):
                                path = filenames.append( path )
        return filenames

def get_filelist(pathname,type):
	if(os.path.isdir(pathname)):
		return get_files(pathname,type)
	else:
		return open(pathname).readlines()

def compress_file (in_file, mode='bz2'):
        try:
                in_data = open(in_file, "rb").read()
                if mode == 'bz2':
                        gzf = bz2.BZ2File (in_file+'.bz2', 'wb')
                else:
                        gzf = gzip.open   (in_file+'.gz',  'wb')
                gzf.write(in_data)
                gzf.close()

        except Exception as e:
                print("Something went wrong compressing ", in_file)

        else:
                os.remove (in_file)


# ---------------------------------------------------
#   Feature File I/O
# ---------------------------------------------------

def writeFtr (output, data):
        if output.endswith('bz2'):
                fout = bz2.BZ2File (output, 'wb')
        else:
                fout = gzip.open   (output, 'wb')
        for frame in range(numpy.shape(data)[0]):
                outputline = ""
                for index in range(numpy.shape(data)[1]):
                        outputline += "%d:%g " % (index+1, data[frame,index])
                fout.write(outputline.strip()+'\n')
        fout.close()

def outputList (output, max_dict, min_dict):
        fout = open(output, 'w')
        index = 1
        for val1, val2 in zip (max_dict, min_dict):
                fout.write("%d %g %g\n" % (index, val1, val2))
                index += 1
        fout.close()


# ---------------------------------------------------
#   Queue and Parallelization
# ---------------------------------------------------

def qsub(script):
	print(script)
	script = os.path.abspath(script)
	rootdir = os.path.dirname(script)
	queue = 'standard'
	QSUB="qsub -j eo -S /bin/bash -o "+rootdir+" -l nodes=1:ppn=1 -q "+ queue +" -d " + rootdir + " " +script
	print("Executing "+QSUB)
	os.system(QSUB)
	return


def dispatchParMap (function, list, *args):
        if os.environ.get('PBS_NUM_PPN') is None:
                print('Simulating ParMap: Len=', len (list))
                return [function (wavFile, *args) for wavFile in list]
        else:
                np = int(os.environ.get('PBS_NUM_PPN'))
                print('Executing ParMap: Len=', len (list), 'NP=', np)
                return parmap.map (function, list, *args, processes=np)
