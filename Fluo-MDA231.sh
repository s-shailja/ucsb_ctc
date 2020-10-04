#!/bin/bash

#################################################
#
# ./Fluo-MDA231.sh indir outdir
# performs segmentation of all tif stacks in <indir> and saves the label masks in <outdir> 
#
# example usage
#
# ./Fluo-MDA231.sh ../Fluo-MDA231/01 ../Fluo-MDA231/01_RES_SEG ../Fluo-MDA231/01_RES
#
#
#################################################

python3 predict_stacks.py -i $1 -os $2 -ot $3
