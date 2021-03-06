#!/bin/bash
# runYuniSegs.sh

# run Yunitator with hard coded models & configs found here and in /vagrant
# assumes Python environment in /home/${user}/

# sane options; fix need to hit control-C multiple times to actually stop
set -o errexit -o pipefail -o noclobber -o nounset

# Absolute path to this script. /home/user/bin/foo.sh
SCRIPT=$(readlink -f $0)
# Absolute path this script is in. /home/user/bin
BASEDIR=`dirname $SCRIPT`

if [ $# -ne 2 ] && [ $# -ne 3 ]; then
  echo "Usage: $0 <audiofile> <RTTM segmentfile> (SkipSIL)"
  echo "       run Yunitator iteratively with precomputed segments"
  echo "       if the 3rd argument is "SkipSIL" do not process input"
  echo "       segments already labeled as silence"
  exit 1;
fi

# test for "SkipSIL" option
SkipSIL=false
if [ $# = 3 ]; then
  if [ $3 == "SkipSIL" ]; then
  SkipSIL=true
  fi
fi


filename=$(basename "$1")
dirname=$(dirname "$1")
extension="${filename##*.}"
basename="${filename%.*}"

# create output file
outfile=$dirname/$basename.yuniSeg.rttm
rm -f $outfile
touch $outfile

# this is set in user's login .bashrc
export PATH=/home/${USER}/anaconda/bin:$PATH


# let's get our bearings: set CWD to path of this script
cd $BASEDIR

# make output folder for features, below input folder
mkdir -p $dirname/Yunitemp/

# make folder for WAV audio segments
segments=$dirname/Yunitemp/segments
mkdir -p $segments

# Iterate over each segment found in the input RTTM (field 4 start 5 duration)
# making segment sized WAV files with sox
# then feeding them to Yunitator

#iterate line by line through input SAD RTTM
while read line; do
  col1=`echo $line | awk '{print $1}'`
  col2=`echo $line | awk '{print $2}'`
  col3=`echo $line | awk '{print $3}'`
  start=`echo $line | awk '{print $4}'`
  dur=`echo $line | awk '{print $5}'`
  col6=`echo $line | awk '{print $6}'`
  col7=`echo $line | awk '{print $7}'`
  col9=`echo $line | awk '{print $9}'`
  segname=$start-$dur

  if (( $(echo "$dur > 0" |bc -l) )); then # skip 0 duration segments
      value=`echo $line | awk '{print $8}'`
      if [ "$value" = "SIL" ] && $SkipSIL ; then # Output SIL segment
	  echo -e $col1"\t"$col2"\t"$col3"\t"$start"\t"$dur"\t"$col6"\t"$col7"\tSIL\t"$col9"\t<NA>" >> $outfile
      else
	  tempfile=$segments/${start}-${dur}.wav
	  sox $1 $tempfile trim $start $dur

	  tfilename=$(basename "$tempfile")
	  tbasename="${tfilename%.*}"

	  # first features
	  ./extract-htk-vm2.sh $tempfile >& /dev/null

	  # then confidences
	  python diarize.py $segments/Yunitemp/$tbasename.htk $segments/Yunitemp/$tbasename.rttm.sorted
	  sort -V -k3 $segments/Yunitemp/$tbasename.rttm.sorted | head -n -1 >| $segments/Yunitemp/$tbasename.rttm

	  segClass=`python maxClass.py $segments/Yunitemp/$tbasename.rttm`
	  echo -e $col1"\t"$col2"\t"$col3"\t"$start"\t"$dur"\t"$col6"\t"$col7"\t"$segClass"\t"$col9"\t<NA>" >> $outfile
      fi
  fi
done < $2

# maybe delete the segments, yes? audio files can get big
rm -rf $segments/*
