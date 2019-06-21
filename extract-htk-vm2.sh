#!/bin/bash

# Use OpenSMILE 2.1.0 only

if [ $# -lt 1 ]; then
  echo "USAGE: extract-htk.sh <INPUT FILE> <TEMP FOLDER>"
  exit 1
fi

filename=$(basename "$1")
dirname=$(dirname "$1")
extension="${filename##*.}"
basename="${filename%.*}"

FEATURE_NAME=med # arbitrary - just a name
INPUT=$1
TEMP=$2
CONFIG_FILE=MED_2s_100ms_htk.conf
OUTPUT_DIR=$dirname/$TEMP/
OPENSMILE=SMILExtract

mkdir -p $OUTPUT_DIR
file=$INPUT
id=`basename $file`
echo "Extracting features for $id ..."
id=${id%.wav}

nosplit=1200
duration=`soxi -D $file|awk '{print int($1)}'`
if [ $duration -gt $nosplit ]; then
    # If the audio file is longer than about 15 minutes, let's
    # split it up to avoid running our of memory
    split=0
    while [ $split -lt $duration ]; do
	echo Now at $split of $duration ...
	LD_LIBRARY_PATH=/home/vagrant/usr/local/lib \
	    $OPENSMILE \
	    -C <(awk -v s=$split '/append=/ {if (s==0) {print ("append=0")} else {print ("append=1")}}; !/append=/ {print $0}' $CONFIG_FILE) \
	    -I <(sox -q $file -t wav - trim $split $nosplit) \
	    -O $OUTPUT_DIR/${id}.htk \
	    -logfile extract-htk.log
	split=`echo $split+$nosplit|bc`
    done

else
    # This is the original code, we use it for shorter utterances

# Use OpenSMILE 2.1.0  
LD_LIBRARY_PATH=/home/vagrant/usr/local/lib \
    $OPENSMILE \
    -C $CONFIG_FILE \
    -I $file \
    -O $OUTPUT_DIR/${id}.htk \
    -logfile extract-htk.log # \
    #>& /dev/null

fi

echo "DONE!"
