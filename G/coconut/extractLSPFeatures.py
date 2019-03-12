#!/opt/python27/bin/python

#export PYTHONPATH=.:/home/iyu/sfep/asr_noisemes/base/lib/python2.7/site-packages

# for convenience
import sys
import os
#import atexit
import tempfile
#import socket
#import time
#import datetime
#import resource
#import __main__ as main
import argparse

# for 'math' stuff
import itertools
import traceback
import numpy

import coconut
#from coconut import opensmile


def help():
        argParser = argparse.ArgumentParser(description='Extracts large scale pooling features')
        argParser.add_argument('input',help='{wav|feature}{dir|list}: data to load')
        argParser.add_argument('outpath',help='output path for the features')
        argParser.add_argument('featIndex',help='index of selected features file')
        argParser.add_argument('--meanvar',help='path to meanvar file',default=None)
        argParser.add_argument('--eigVal',help='path to eigVal file',default='')
        argParser.add_argument('--eigVec',help='path to eigVec file',default='')
        argParser.add_argument('--comps',nargs='+',type=int,
                               help='Space separated list of #components(LSPF dimensions). Defaults to [100,300].')
        argParser.add_argument('--smilepath',help='path to SMILExtract',
                               default='/home/iyu/sfep/asr_noisemes/libraries/openSmile/bin/SMILExtract')
        argParser.add_argument('--config',help='path to OpenSmile configuration',
                               default='/home/iyu/sfep/LargeScaleAudioPoolingFeatures/config/OpenSmile/MED_2s_100ms_csv.conf')
        argParser.add_argument('--tmp',help='path to temporary files',default='/scratch')
        argParser.add_argument('--save',dest='save',action='store_true')
        argParser.add_argument('--no-save',dest='save',action='store_false')
        argParser.set_defaults(save=False)
        return argParser.parse_args()


def getFeatures (inFile, outpath, column_indices, config, smile, scratch, save):
        try:
                inFile     = inFile.rstrip("\n")
                baseName   = os.path.basename(inFile)
                (vid, lab) = os.path.splitext(baseName)

                # find out what file to read ('tmp') and do it
                if   lab == ".raw":
                        tmp = inFile
                elif lab == ".bz2":
                        tmp = inFile
                        (vid, dummy) = os.path.splitext (vid)
                elif lab == ".wav":
                        tmp = os.path.join (scratch, vid+'.raw')
                        res = coconut.opensmile.extractFeatures (inFile, tmp, config)
                        if res[0] or not os.path.getsize (tmp):
                                print 'getFeatures: something is fishy0 in', vid, '>>>', res[2]
                                return
                else:
                        wav = os.path.join (scratch, vid+'.wav')
                        res = coconut.opensmile.extractWav (inFile, wav)
                        if res[0] or not os.path.getsize (wav):
                                print 'getFeatures: something is fishy1 in', vid, '>>>', res[2]
                                return
                        tmp = os.path.join (scratch, vid+'.raw')
                        res = coconut.opensmile.extractFeatures (wav, tmp, config)
                        if res[0] or not os.path.getsize (tmp):
                                print 'getFeatures: something is fishy2 in', vid, '>>>', res[2]
                                return
                        os.remove (wav)

                filteredFeatures = coconut.utils.filterColumns (tmp, column_indices)

                # compress or remove the feature file
                if tmp == inFile:
                        pass
                elif save:
                        coconut.utils.compress_file (tmp)
                else:
                        os.remove (tmp)

                # some checks
                if filteredFeatures.ndim == 1:
                       print 'getFeatures: something is fishy3 in', vid
                       return

                # statistics
                len   = numpy.shape (filteredFeatures)[0]
                min   = numpy.amin  (filteredFeatures,    axis=0)
                max   = numpy.amax  (filteredFeatures,    axis=0)
                sum   = numpy.sum   (filteredFeatures,    axis=0)
                smosq = numpy.sum   (filteredFeatures**2, axis=0)

                print '  Thread=', os.getpid(), 'Key=', vid, 'Frames=', len
                if not len or numpy.allclose (min, max):
                        print 'getFeatures: something is fishy4 in', vid
                        return 

        except Exception as e:
                print ('getFeatures: caught exception in worker thread:')
                traceback.print_exc()
                print()
                raise e
                
        else:
                return vid, filteredFeatures, long(len), min, max, sum, smosq


def extractNoisemesPar (wavFile, outpath, column_indices, config, smile, scratch, save):

        (vid, filteredFeatures, len, min, max, sum, smosq) = getFeatures (wavFile, outpath, 
                                                                          column_indices, config,
                                                                          smile, scratch, save)

        featSelectFile = os.path.join (outpath, vid+".mat.bz2")
        #numpy.savetxt (featSelectFile, filteredFeatures, fmt='%.4g')
        coconut.utils.writeFtr (featSelectFile, filteredFeatures)

        return len, sum, smosq, min, max


def extractLSPFPar (wavFile, outpath, column_indices, meanvar, eigVec, eigVal, n_comps, 
                    config, smile, scratch, save):

        (vid, filteredFeatures, len, min, max, sum, smosq) = getFeatures (wavFile, outpath, 
                                                                          column_indices, config,
                                                                          smile, scratch, save)
 
        standardFeatures = coconut.utils.standardize (filteredFeatures, meanvar[0], meanvar[1])
        for comp in n_comps:
                pcaFile = os.path.join (outpath, vid+".pca"+str(comp))
                pcaFeat = coconut.utils.pcaTransform (standardFeatures, eigVal, eigVec, n_comps=comp)
                numpy.savetxt (pcaFile, pcaFeat, fmt='%1.4f')

        return len, sum, smosq, min, max


def main(args):
        wavRoot = os.path.abspath(args.input)
	print "Loading files from "+wavRoot
	wavFileSet = coconut.utils.get_filelist(wavRoot,"wav")
	config = os.path.abspath(args.config)
	outFilePath = os.path.abspath(args.outpath)
	SMILE = os.path.abspath(args.smilepath)
	featIndex = os.path.abspath(args.featIndex)

        print "Loading configuration from "+featIndex
        if len(open(featIndex).readline().split(' ')) == 2:
                # alternatively - seems to be in format used by Lara (and Yipei?)
                column_indices = sorted([int(line.strip().split(' ')[1])-1 for line in open(featIndex)])
        else:
                # convert matlab indices to python indices (Shourabh?)
                column_indices = [int(colid)-1 for colid in open(featIndex).readline().rstrip("\n").split(",")]

        if args.meanvar:
                print "Loading the mean var Index "+args.meanvar
                meanvar = numpy.loadtxt(os.path.abspath(args.meanvar))
	
                #if args.eigVal and args.eigVec:
                print "Loading the principal components "+args.eigVal,args.eigVec
                eigVal = numpy.loadtxt(os.path.abspath(args.eigVal))
                eigVec = numpy.loadtxt(os.path.abspath(args.eigVec),delimiter=',')
        
                #if args.comps:
                n_comps = args.comps
                print "Components Extracted: "+str(n_comps)

        if args.save:
                tmpPath = args.tmp
        else:
                tmpPath = tempfile.mkdtemp (dir=args.tmp)

        if not os.path.exists(outFilePath):
                os.makedirs(outFilePath)
        if not os.path.exists(tmpPath):
                os.makedirs(tmpPath)

        # now do 'map' in parallel
        print 'TmpPath=', tmpPath, 'Save=', args.save
        if args.meanvar:
                mapresult = coconut.utils.dispatchParMap (extractLSPFPar, wavFileSet,
                                                          outFilePath, column_indices, 
                                                          meanvar, eigVec, eigVal, n_comps, 
                                                          config, SMILE, tmpPath, args.save) 
        else:
                mapresult = coconut.utils.dispatchParMap (extractNoisemesPar, wavFileSet, 
                                                          outFilePath, column_indices, 
                                                          config, SMILE, tmpPath, args.save)
                
        # the 'reduce' step
        L1 = len (mapresult)
        mapresult = filter (lambda x: x!=None, mapresult)
        L2 = len (mapresult)
        print 'Done with parmap,', L2, 'of', L1, 'keys processed'        

        tlen, tsum, tsmosq, tmin, tmax = mapresult[0]
        for (rlen, rsum, rsmosq, rmin, rmax) in mapresult[1:]:
                tlen   += rlen
                tsum   += rsum
                tsmosq += rsmosq
                tmin    = numpy.amin ((tmin, rmin), axis=0)
                tmax    = numpy.amax ((tmax, rmax), axis=0)
        mean = tsum/tlen
        msos = tsmosq/tlen

        # info
        print "Elements=", len(mapresult)
        print "Length=", tlen
        #print "Minimum=", tmin
        #print "Mean=", mean
        #print "Maximum=", tmax

        # results
        numpy.savetxt            (os.path.join (outFilePath, 'mean'), mean)
        numpy.savetxt            (os.path.join (outFilePath, 'sumofsquares'), msos)
        numpy.savetxt            (os.path.join (outFilePath, 'minimum'), tmin)
        numpy.savetxt            (os.path.join (outFilePath, 'maximum'), tmax)
        coconut.utils.outputList (os.path.join (outFilePath, 'scale.txt'), tmax, tmin)
        with open                (os.path.join (outFilePath, 'length'), 'w') as the_file:
                the_file.write('{:d} {:d}\n'.format(len(mapresult),tlen))

        # cleanup
        if not args.save:
                os.rmdir (tmpPath)


# call the main function & parse the arguments
if __name__ == '__main__':
        main(help())
