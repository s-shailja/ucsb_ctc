# Semi-supervised segmentation and tracking algorithms for [cell segmentation](http://celltrackingchallenge.net/)

We present a novel weakly supervised 3D nuclei segmentation method that consists of deep learning based nuclei detection, watershed segmentation, and a boundary correction algorithm using supervoxels. Additionally, we present a simple and efficient graph-based tracking algorithm utilizing relative nuclei location information extracted from the adjacency graph. 


For more details about our methodology, please refer to our [paper](https://arxiv.org/abs/2010.13343).

The performance of our proposed method on CTC 2020 dataset is shown in the following table:

|Dataset|DET|SEG|TRA|OP_CSB|OP_CTB|
|---|---|---|---|---|---|
|Fluo-N3DH-CE|0.927|0.705|0.895|0.816|0.800|
|Fluo-C3DL-MDA231|0.839|0.545|0.795|0.692|0.670|

## Citation

The system was employed for our research presented in [1], where we propose a novel semi supervised nuclei segmentation method utilizing Simple linear Iterative Clustering (SLIC) boundary adherence and a graph-based tracking algorithm utilizing relative cell location information. If the use of the software or the idea of the paper positively influences your endeavours, please cite [1].

[1] **S. Shailja**, Jiaxiang Jiang, and B.S. Manjunath, "[Semi supervised segmentation and graph-based tracking of 3D nuclei in time-lapse microscopy.](https://arxiv.org/abs/2010.13343)"  Submitted to IEEE ISBI 2021.

## How to run

The command 

`./Fluo-MDA231.sh indir outdirseg outdirtrack datatype`

runs the segmentation and tracking pipeline on all tif stacks in `indir` and saves the label masks in `outdirseg` and `outdirtrack` respectively. The dataset can be passed through datatype argument.


## Example usage


`./Fluo.sh ./01 ./01_RES_SEG ./01_RES_TRACK "N3DCHCE`


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
