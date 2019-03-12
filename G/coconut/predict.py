#!/opt/python27/bin/python2.7

import sys, os, copy, argparse, gzip
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import traceback, parmap, itertools, time
import os.path as path
from collections import defaultdict

# predict.py
# Takes in mat files which were created from OpenSmile
# Does rescaling, predicting, and bag-of-feature generation
# Outputs BOF files

def help():
        argParser = argparse.ArgumentParser(description='Makes predictions on LSPFs')
        argParser.add_argument('modellist',help='list of models')
        argParser.add_argument('testlist',help='files to predict')
        argParser.add_argument('outputPath',help='output path')
        argParser.add_argument('--keepPredict',help='to keep intermediate prediction files, specify a directory',default='/scratch')
        argParser.add_argument('--scale',help='scale file',default='')
        argParser.add_argument('--kind',help='kind of bof creation',default='ave')
        argParser.add_argument('--seg',help='minimum number of segments allowed in a clip',default=1)
        return argParser.parse_args()

# -------------------------------------
#   deal with scaling of features
# -------------------------------------

def load_param(filename):
        global maxDict, minDict
        # sets globals maxDict/ minDict
        maxDict = {}
        minDict = {}

        print 'Loading scaling parameters', filename
        for line in open(filename):     
                (index, maxVal, minVal) = line.strip().split()
                minDict[int(index)] = float(minVal)
                maxDict[int(index)] = float(maxVal)
        return (maxDict, minDict)

def  CalculateScaled(original, maxValue, minValue, lower=-1.0, upper=1.0):
        if original==maxValue:
                scaledValue = upper
        elif original==minValue:
                scaledValue = lower
        else:
                scaledValue = lower + (upper - lower) * (original - minValue) / (maxValue - minValue)        
        return scaledValue      

# -------------------------------------
#   compute predictions
# -------------------------------------

#support strategy of equal, segment
def SelectSample(testfile, strategy='equal', interval=100):
        #interval unit is msec, we predict on all data now
        #uses global maxDict/ minDict to scale features, if present
	x = np.array([])
	timelist = []

	index=0
	index_interval=int(interval/100)
	first=True
	if ".gz" in testfile:
	 	opened = gzip.open(testfile, 'rb')
	else:
		opened = open(testfile,'r')
	for line in opened.readlines():
		if index % index_interval==0:
			timelist.append(float(index)*100/1000)
	    		xi = []
            		line_list = line.strip().split()
                        if 'maxDict' in globals():
                                for item in line_list:
                                        index = int(item.split(':')[0])
                                        value = CalculateScaled (float(item.split(':')[1]), maxDict[index], minDict[index])
                                        xi.append(float(value))  #index starts from 1
                        else:
                               for item in line_list:
                                        value = float(item.split(':')[1])
                                        xi.append(float(value))  #index starts from 1
            		xi_array = np.array([xi])

            		if first:
                		x = np.append(x, xi_array)
                		x = [x]
                		first = False
            		else:
                		x = np.append(x, xi_array, axis=0)		

		index += 1
	opened.close()
	return x, timelist

def Predict(x, modelbox):
	hyp_labels=[]
	hyp_confidences=[]
	for i in range(0, len(modelbox)):
		m = modelbox[i]
		hyp_labels.append(m.predict(x))
        	hyp_confidences.append(m.predict_proba(x))    
                #can it output label and confidence at the same time?
	return hyp_labels, hyp_confidences

def OutputPredict(outputPath, filename, hyp_label, hyp_conf):
	if outputPath in "":
		out = os.path.join("/scratch",filename)
	else:
		out = os.path.join(outputPath,filename)
	f=gzip.open(out, 'w')
	InsNum = len(hyp_label[0])
	print "predicting", InsNum, "into", out
	for n in range(0, InsNum):
		predictline = ""
		for m in range(0, len(hyp_label)):
			lb = hyp_label[m][n]
			if lb > 0:
				conf = hyp_conf[m][n][1]
			else:
				conf = hyp_conf[m][n][0]
			predictline += "%s:%s " % (lb, conf)
		f.write(predictline + '\n')
	f.close()
	return out


# -------------------------------------
#   compute bof
# -------------------------------------

def CalculateBof(featurelist, kind):
        total = 0
        first = True
        for ftrline in featurelist:
                total += 1
                vector = []
                for item in ftrline.split():
                        (lb, conf) = item.split(':')
                        if float(lb)>0:
                                vector.append(float(conf))
                        else:
                                vector.append(1-float(conf))
                if first:
                        bof = copy.deepcopy(vector)
                        bofsquare = []
                        for i in range(0, len(bof)):
                                bofsquare.append(pow(bof[i], 2))
                        first=False
                else:
                        if kind == "ave":
                                for i in range(0, len(vector)):
                                        bof[i] += vector[i]
                        elif kind=="max":
                                for i in range(0, len(vector)):
                                        bof[i] = max(bof[i], vector[i])
                        elif kind=='avevar':
                                for i in range(0, len(vector)):
                                        bof[i] += vector[i]
                                        bofsquare[i] += pow(vector[i], 2)
                        else:
                                print "kind error:", kind

        if kind=="ave":
                for i in range(0, len(bof)):
                        bof[i] = bof[i]/total

        elif kind=='avevar':
                average = []
                var = []
                for i in range(0, len(bof)):
                        average.append( float(bof[i])/total )
                        bof[i] = average[i]
                last = len(average)
                for i in range(0, len(bofsquare)):
                        var.append( float(bofsquare[i])/total - pow(average[i], 2) )
                        bof.append(var[i])
        return bof

def outputBof(bof, outputfile):
        outputline = ""
        for i in range(0, len(bof)):
                outputline += "%s:%s " % (i+1, bof[i])
                
        fout = open(outputfile, 'w')
        fout.write(outputline.strip() + '\n')
        fout.close()

def iterateFiles (file, outputPath, modelbox, keepPredict, kind, seg):
        try:
                baseName    = os.path.basename(file)
                (clip, lab) = baseName.split(os.extsep, 1) #separate by first period
                if (not os.path.isfile(file) or not os.path.getsize(file)):
                        print 'Could not find file:', file
                        return
                test_x, timelist = SelectSample(file)
                hyp_label, hyp_conf = Predict(test_x, modelbox)
                outPath = OutputPredict (keepPredict, clip+'.txt.gz', hyp_label, hyp_conf)
		
		ftrpool=[line.strip() for line in gzip.open(outPath, 'r')]
	        timeLen = len(ftrpool)
#		print clip
	        if timeLen<seg:
	                print "Clip length is less than the segment number. Skipping: "+clip
	        else:
	                step = int(float(timeLen)/seg)
	                bof = []
	                for i in range(0, seg):
	                        bof += CalculateBof(ftrpool[i*step:(i+1)*step], kind)
	                outputfile = path.join(outputPath, clip+'.bof')
	                outputBof(bof, outputfile)
	                print "BOF complete:", clip
	        if keepPredict in "/scratch":
	                print "Removing predict:",clip
	                os.remove(path.join("/scratch", clip+".txt.gz"))
        except Exception, e:
                # Something went wrong
                print 'Caught exception in worker thread:'
                traceback.print_exc()
                raise e
        else:
                # Everything is ok
                return file

# -------------------------------------
#   main loop
# -------------------------------------

def main(args):
        modellist  = args.modellist
        testlist   = [line.strip() for line in open(args.testlist)] #paths and file names
	outputPath = args.outputPath
	keepPredict= args.keepPredict	
        scale      = args.scale
	kind       = args.kind
	seg        = int(args.seg)

        modelbox=[]
        model_name_list=[]
        for line in open(modellist):
                modelfile = line.strip()
                #print "Loading...", modelfile
                model = joblib.load(modelfile)
                modelbox.append(model)
                #not sure whether the model type can be add to the list
        print "Loaded model", modellist

        if 'scale' in locals() and len(scale):
                (maxDict, minDict) = load_param(scale)

        # now do 'map' in parallel
        print 'Executing predict parmap:', len (testlist)
        if os.environ.get('PBS_NUM_PPN') is None:
                mapresult = [iterateFiles (file, outputPath, modelbox, keepPredict, kind, seg) for file in testlist]
        else:
                np = int(os.environ.get('PBS_NUM_PPN'))
                print '  np=', np
                mapresult = parmap.map (iterateFiles, testlist, outputPath, modelbox, keepPredict, kind, seg, processes=np)
        print 'Done!'


if __name__ == '__main__':
        main(help())
