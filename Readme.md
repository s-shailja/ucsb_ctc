
## Overview


The command 

`./Fluo-MDA231.sh indir outdirseg outdirtrack`

runs the segmentation and tracking pipeline on all tif stacks in `indir` and saves the label masks in `outdirseg` and `outdirtrack` respectively.


## Example usage


`./Fluo-MDA231.sh ./01 ./01_RES_SEG ./01_RES_TRACK`


```
./01/
├── t003.tif
├── t008.tif
├── t013.tif
├── t018.tif
├── t023.tif
├── t028.tif

./01_RES_SEG/
├── mask003.tif
├── mask008.tif
├── mask013.tif
├── mask018.tif
├── mask023.tif
├── mask028.tif

./01_RES_TRACK/
├── res_track.txt
├── mask008.tif
├── mask013.tif
├── mask018.tif
├── mask023.tif
├── mask028.tif
```
