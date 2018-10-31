#!/bin/bash
# runDiarNoisemes.sh

# run OpenSAT with hard coded models & configs found here and in /vagrant
# assumes Python environment in /home/${user}/
# usage: runDiarNoisemes.sh <folder containing .wav files to process>

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
  echo "Usage: runYunitator.sh <audiofile>"
  exit 1;
fi

# let's get our bearings: set CWD to path of this script
cd $BASEDIR

# make output folder for features, below input folder
mkdir -p $dirname/Yunitemp/

# first features
./extract-htk-vm2.sh $1

# then confidences
python diarize.py $dirname/Yunitemp/$basename.htk $dirname/Yunitemp/$basename.rttm_sorted
sort -V -k3 $dirname/Yunitemp/$basename.rttm_sorted > $dirname/Yunitemp/$basename.rttm


