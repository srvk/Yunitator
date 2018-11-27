#!/bin/bash
# runClasses.sh

# run OpenSAT with hard coded models & configs found here and in /vagrant
# assumes Python environment in /home/${user}/
# usage: runClasses.sh <folder containing .wav files to process>
# produces RTTM format with class labels found in noisemeclasses.txt

# Absolute path to this script. /home/user/bin/foo.sh
SCRIPT=$(readlink -f $0)
# Absolute path this script is in. /home/user/bin
BASEDIR=`dirname $SCRIPT`


filename=$(basename "$1")
dirname=$(dirname "$1")
extension="${filename##*.}"
basename="${filename%.*}"

# this is set in user's login .bashrc
export PATH=/home/${USER}/anaconda/bin:$PATH

if [ $# -ne 1 ]; then
  echo "Usage: $0 <audiofile>"
  exit 1;
fi

# let's get our bearings: set CWD to path of this script
cd $BASEDIR

# make output folder for features, below input folder
mkdir -p $1/feature

# first features
for file in `ls $1/*.wav`; do
  SSSF/code/feature/extract-htk-vm2.sh $file
done

# then confidences
python SSSF/code/predict/1-confidence-vm3.py $1
#python SSSF/code/predict/1-confidence-vm5.py $1

