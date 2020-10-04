import numpy as np
import os
from glob import glob
from tifffile import imread, imsave
from itertools import product
import re
import argparse
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries
from keras.models import load_model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import pdb
from keras.models import load_model
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from skimage.segmentation import slic
from track import track_main
import tensorflow as tf
np.random.seed(42)

def normalize(x, pmin=3, pmax=99.8, axis = None, clip=False):
    mi = np.percentile(x, pmin, axis = axis, keepdims=True).astype(np.float32)
    ma = np.percentile(x, pmax, axis = axis, keepdims=True).astype(np.float32)
    x = x.astype(np.float32)
    eps = 1.e-20
    y = (1. * x - mi) / (ma - mi+eps)
    if clip:
        y = np.clip(y, 0, 1)
    return y

def dice_coef(y_true,y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true*y_pred))
    return (2*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def dice_loss(y_true,y_pred):
    return 1-dice_coef(y_true,y_pred)


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto( intra_op_parallelism_threads=22, 
                        inter_op_parallelism_threads=22, 
                        allow_soft_placement=True)
    config.gpu_options.allow_growth=True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-i', "--indir",
                        default="/projects/CSBDeep/ISBI/upload_test/01")

    parser.add_argument('-os', "--outdirseg",
                        default="/home/tom/ctc/01_RES")

    parser.add_argument('-ot', "--outdirtrack",
                        default="/home/tom/ctc/01_TRACK")

    parser.add_argument('--threshold_abs',type = float,
                        default=.3)
    parser.add_argument('--min_distance',type = int,
                        default=4)

    parser.add_argument('--tile_size', type = int, nargs = 3,
                        default=[-1,512,256])

    args = parser.parse_args()

    model_center = load_model("models/center_cell.h5", custom_objects={'dice_loss':                   
dice_loss, 'dice_coef': dice_coef})
    model_mask = load_model("models/prob_cell.h5",custom_objects={'dice_loss':                   
dice_loss, 'dice_coef' : dice_coef})

    files = sorted(glob(os.path.join(args.indir,"*.tif")))
    for i,f in enumerate(files):
        print("predicting %s/%s"%(i+1,len(files)) )
        img = imread(f)
        img_new = np.zeros((32,512,512))
        img_new[0:30] = img
        img_inputs = np.expand_dims(img_new,3)
        img_inputs = np.expand_dims(img_inputs,0)
        u_center = model_center.predict(img_inputs)
        u_mask = model_mask.predict(img_inputs)
        u_mask = np.reshape(u_mask[0][:30], (30,512,512))
        u_center = np.reshape(u_center[0][:30], (30,512,512))
        center = u_center > 0.3
        peaks = peak_local_max(center, min_distance= args.min_distance, threshold_abs=args.threshold_abs, indices=False, exclude_border=1)
        # print(peaks.shape)
        markers = ndi.label(peaks)[0]
        mask = u_mask>.8
        # labels = watershed(-u_mask, markers, mask=mask)

        ref_label = watershed(-u_mask, markers, mask=mask)
        
        segments = slic(img, n_segments=1400, compactness=0.1, multichannel = False, sigma = 0)
        ref_label_x, ref_label_y, ref_label_z = segments.shape
        if args.outdirseg and len(args.outdirseg)>0:
            os.makedirs(args.outdirseg, exist_ok = True)
            tp = re.findall("t([0-9]+?)\.",os.path.basename(f))
            if len(tp)==0:
                tp = os.path.basename(f)
            else:
                tp = tp[0]

            labels = (np.array(segments, dtype=np.int32)).reshape(ref_label.shape)

            final_labels = np.empty([ref_label_x,ref_label_y,ref_label_z])
            dictionary_labels = {}
            for l in range (ref_label_x):
                for m in range(ref_label_y):
                    for n in range(ref_label_z):
                        if not dictionary_labels.get(labels[l][m][n]):
                            dictionary_labels[labels[l][m][n]] = [ref_label[l][m][n]]
                        else:                                
                            dictionary_labels[labels[l][m][n]].append(ref_label[l][m][n])
            for l in dictionary_labels.keys():
                L = dictionary_labels[l]
                d = {}
                
                for i in L:
                    if not d.get(i):
                        d[i] = 1
                    else:
                        d[i] += 1
                ma_index = 0
                ma_value = 0
                for i in d.keys():
                    if (d[i] > ma_value):
                        ma_index = i
                        ma_value = d[i]

                dictionary_labels[l] = ma_index

            for l in range (ref_label_x):
                for m in range(ref_label_y):
                    for n in range(ref_label_z):
                        final_labels[l][m][n] = dictionary_labels[labels[l][m][n]] 
                        if (ref_label[l][m][n] == 0):
                            ref_label[l][m][n] = final_labels[l][m][n]

            imsave(os.path.join(args.outdirseg,"mask%s.tif"%tp),ref_label.astype(np.uint16))
    track_main(args.outdirseg,args.outdirtrack) 



