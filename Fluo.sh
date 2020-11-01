#!/bin/bash

#################################################
#
# ./Fluo.sh indir outdir
# performs segmentation of all tif stacks in <indir> and saves the label masks in <outdir> 
#
# example usage
#
# ./Fluo.sh ../Fluo-N3DH-CE/01 ../Fluo-N3DH-CE/01_RES_SEG ../Fluo-N3DHCE/01_RES "N3DHCE"
#
# ./Fluo.sh ../Fluo-MDA231/01 ../Fluo-MDA231/01_RES_SEG/ ../Fluo-MDA231/01_RES/ "MDA231"
#################################################

python3 predict_stacks.py -i $1 -os $2 -ot $3 -dt $4
