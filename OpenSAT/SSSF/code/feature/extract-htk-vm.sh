# Use OpenSMILE 2.1.0 only

if [ $# -lt 1 ]; then
  echo "USAGE: extract-htk.sh <INPUT FILE>"
  exit 1
fi

FEATURE_NAME=med # arbitrary - just a name
INPUT=$1
CONFIG_FILE=MED_2s_100ms_htk.conf
OUTPUT_DIR=/vagrant/data/feature/evl.${FEATURE_NAME}.htk
OPENSMILE=SMILExtract

mkdir -p $OUTPUT_DIR
file=$INPUT
id=`basename $file`
echo "Extracting features for $id ..."
id=${id%.wav}

# Use OpenSMILE 2.1.0  
LD_LIBRARY_PATH=/home/vagrant/usr/local/lib \
    $OPENSMILE \
    -C $CONFIG_FILE \
    -I $file \
    -O $OUTPUT_DIR/${id}.htk \
    -logfile extract-htk.log # \
    #>& /dev/null

echo "DONE!"
