#!/bin/bash

SCRIPTDIR="$( cd "$(dirname "$0")" ; pwd -P )"
COOKIE="$SCRIPTDIR/cookie"
FILE_ID="12cqKTPtQBAu780219hEbST7VwQuf6xDH"
FILENAME="$SCRIPTDIR/ssd512-hollywood-trainval-bs_16-lr_1e-05-scale_pascal-epoch-187-py3.5.h5"

curl -c $COOKIE -s -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
curl -Lb $COOKIE "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' $COOKIE`&id=${FILE_ID}" -o $FILENAME && rm -f $COOKIE
