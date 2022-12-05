#!/bin/bash
# Date: 2022-12-3
# Author: Create by zjh
# Description: This script is to preprocess dataset files
# Arguments: dataset name; dataset files dir; save dir;
# Version: 0.1

DATASET_NAME=$1
FROM_DIR=$2
TO_DIR=$3
JOERN=`whereis joern | awk -F '[:]' '{print $2}'`/joern-cli/joern
JOERN_PARSE=`whereis joern | awk -F '[:]' '{print $2}'`/joern-cli/joern-parse
EXTERNAL_PATH=$(cd `dirname $0`; cd ..; pwd)/joern/files/sensi_funcs.txt
SCALA_FUNCS_PATH=$(cd `dirname $0`; cd ..; pwd)/joern/joern-cli/scripts/extract-funcs-info.sc
SCALA_POINTS_PATH=$(cd `dirname $0`; cd ..; pwd)/joern/joern-cli/scripts/get-points.sc

#echo $FROM_DIR
#echo $TO_DIR
#echo $JOERN
#echo $JOERN_PARSE
#echo $EXTERNAL_PATH
#echo $SCALA_FUNCS_PATH
#echo $SCALA_POINTS_PATH

# Parse C files in $FROM_DIR by $JOERN_PARSE and save to $TO_DIR/parse_$DATASET_NAME
mkdir $TO_DIR/parse_$DATASET_NAME
for file in `ls $FROM_DIR`
do
  if [ -d $FROM_DIR/$file ]
  then
    bash $JOERN_PARSE $FROM_DIR/$file -o $TO_DIR/parse_$DATASET_NAME/$file.bin
#    echo $FROM_DIR/$file
#    echo $TO_DIR/parse_$DATASET_NAME/$file.bin
  fi
done

# Call $JOERN to extract function cpgs and points info then save to $TO_DIR/results_$DATASET_NAME
#bash $JOERN | "importCpg(\"$TO_DIR/parse_$DATASET_NAME/group0.bin\")"