#! /usr/bin/python 

import subprocess
import os

def extractFeatures (wavFile, outFile, config, cmd='/home/iyu/sfep/asr_noisemes/libraries/openSmile/bin/SMILExtract'):
        p = subprocess.Popen ([cmd, '-C', config, '-I', wavFile, '-O', outFile], \
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
	return (p.returncode, stdout, stderr)

def extractWav (mp4File, wavFile, cmd='ffmpeg', clean=1):
        p = subprocess.Popen ([cmd, '-y', '-i', mp4File, '-ac', '1', '-ar', '16000', wavFile], \
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()

        # this is necessary because opensmile is stupid and chokes on ffmpeg's audio files
        if not p.returncode and clean:
                q = cleanWav (wavFile, wavFile + '.wav')
                if not q[0]:
                        os.rename (wavFile + '.wav', wavFile)

                return q
                        
	return (p.returncode, stdout, stderr)

def cleanWav (mp4File, wavFile, cmd='sox'):
        p = subprocess.Popen ([cmd, mp4File, '-c', '1', wavFile], \
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = p.communicate()
	return (p.returncode, stdout, stderr)
