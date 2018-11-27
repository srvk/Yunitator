#!/bin/bash
# runDiarNoisemes.sh

# run OpenSAT with hard coded models & configs found here and in /vagrant
# assumes Python environment in /home/${user}/
# usage: runDiarNoisemes.sh <folder containing .wav files to process>

# Absolute path to this script. /home/user/bin/foo.sh
SCRIPT=$(readlink -f $0)
# Absolute path this script is in. /home/user/bin
BASEDIR=`dirname $SCRIPT`

# this is set in user's login .bashrc

if [ $# -ne 1 ]; then
  echo "Usage: runYunitator.sh <audiofile>"
  exit 1;
fi

# let's get our bearings: set CWD to path of this script
cd $BASEDIR

# make output folder for features, below input folder
mkdir -p $1/feature/

for f in `ls $1/*.wav`; do
    ./extract-htk-vm2.sh $f
done

python yunified.py yunitator $1 2000
for f in `ls $1/feature/*.rttm.sorted`; do
    filename=$(basename "$f")
    basename="${filename%.*}"
    
    sort -V -k3 $f > $1/feature/$basename   
done
