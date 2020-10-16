import os
import numpy as np
from glob import glob
from skimage.io import imread,imsave
#from tifffile import imread,imsave
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import pdb

class cell_data(keras.utils.Sequence):
    def __init__(self,input_path,gt_path):
        #self.batch_size = batch_size
        #self.img_size = img_size
        self.input_path = input_path
        self.gt_path = os.path.join(gt_path,'TRA')
        #self.all_gts = os.listdir(self.gt_path)
    def __len__(self):
        return len(os.listdir(self.gt_path))-1
    def __getitem__(self,idx):
        gt_file = "man_track"+format(idx,'03d')+'.tif'
        #gt_file_noext,_ = os.path.splitext(gt_file)
        #gt_file_tmp = gt_file.split('_')
        #time = gt_file_tmp[2]
        #z_slice = gt_file_tmp[3]
        #print (gt_file)
        #pdb.set_trace()
        gt = imread(os.path.join(self.gt_path,gt_file))
        gt_b = np.where(gt>0,1,0)
        img_file = 't'+format(idx,'03d')+'.tif'
        img = imread(os.path.join(self.input_path,img_file))
        img = img/2**8
        img_new = np.zeros((32,512,512))
        img_new[0:30] = img
        gt_b_new = np.zeros((32,512,512))
        gt_b_new[0:30] = gt
        img_new = np.expand_dims(img_new,3)
        gt_b_new = np.expand_dims(gt_b_new,3)
        img_new = img_new[None,:,:,:,:]
        gt_b_new = gt_b_new[None,:,:,:,:]
        #img_slice = img[int(z_slice)]
        return img_new,gt_b_new


def get_model(img_size):
    num_classes = 1
    inputs = keras.Input(shape = img_size)
    conv1 = layers.Conv3D(8,3,activation='relu',padding="same",data_format="channels_last")(inputs)
    conv1 = layers.Conv3D(8,3,activation='relu',padding="same")(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = layers.Conv3D(16,3,activation='relu',padding="same")(pool1)
    pool2 = layers.MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = layers.Conv3D(32,3,activation='relu',padding="same")(pool2)
    pool3 = layers.MaxPooling3D(pool_size=(2,2,2))(conv3)

    conv4 = layers.Conv3D(64,3,activation='relu',padding="same")(pool3)
    pool4 = layers.MaxPooling3D(pool_size=(2,2,2))(conv4)


    conv5 =  layers.Conv3D(128,3,activation='relu',padding="same")(pool4)
    conv5 =  layers.Conv3D(128,3,activation='relu',padding="same")(conv5)


    up6 =  layers.Conv3D(64,2,activation='relu',padding="same")(layers.UpSampling3D(2)(conv5))
    merge6 = layers.concatenate([conv4,up6],axis=-1)
    conv6 =  layers.Conv3D(64,3,activation='relu',padding="same")(merge6)
    conv6 =  layers.Conv3D(64,3,activation='relu',padding="same")(conv6)



    up7 =  layers.Conv3D(32,2,activation='relu',padding="same")(layers.UpSampling3D(2)(conv6))
    merge7 = layers.concatenate([conv3,up7],axis=-1)
    conv7 =  layers.Conv3D(32,3,activation='relu',padding="same")(merge7)
    conv7 =  layers.Conv3D(32,3,activation='relu',padding="same")(conv7)


    up8 =  layers.Conv3D(16,2,activation='relu',padding="same")(layers.UpSampling3D(2)(conv7))
    merge8 = layers.concatenate([conv2,up8],axis=-1)
    conv8 =  layers.Conv3D(16,3,activation='relu',padding="same")(merge8)
    conv8 =  layers.Conv3D(16,3,activation='relu',padding="same")(conv8)


    up9 =  layers.Conv3D(8,2,activation='relu',padding="same")(layers.UpSampling3D(2)(conv8))
    merge9 = layers.concatenate([conv1,up9],axis=-1)
    conv9 =  layers.Conv3D(8,3,activation='relu',padding="same")(merge9)
    conv9 =  layers.Conv3D(8,3,activation='relu',padding="same")(conv9)



    '''
    #Downsampling
    for filters in [8,16,32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters,3,padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3D(filters,3,padding="same")(x)
        x = layers.BatchNormalization()(x)
    
        x = layers.MaxPooling3D(3,strides=2,padding="same")(x)

        residual = layers.Conv3D(filters,1,strides=2,padding="same")(previous_block)

        x = layers.add([x,residual])
        previous_block = x

    #Upsampling
    for filters in [32,16,8,4]:
        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters,3,padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv3DTranspose(filters,3,padding="same")(x)
        x = layers.BatchNormalization()(x)
    
        x = layers.UpSampling3D(2)(x)

        residual = layers.UpSampling3D(4)(previous_block)
        residual = layers.Conv3D(filters,1,strides=2,padding="same")(residual)
        x = layers.add([x,residual])
        previous_block = x
    '''
    outputs = layers.Conv3D(num_classes,1,activation="sigmoid")(conv9)
    model = keras.Model(inputs,outputs)
    return model


def dice_coef(y_true,y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true*y_pred))
    return (2*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

def dice_loss(y_true,y_pred):
    return 1-dice_coef(y_true,y_pred)
def train():
    keras.backend.clear_session()
    #model = get_model(img_size)
    #change input and gt pathmode
    train_gen = cell_data('/home/tom/ctc_train/Fluo-C3DL-MDA231/01','/home/tom/ctc_train/Fluo-C3DL-MDA231/01_GT')
    
    model = get_model((32,512,512,1))
    model.compile(optimizer="adam",loss=dice_loss,metrics=[dice_coef])

    callbacks = [keras.callbacks.ModelCheckpoint("ctc_train/center_cell.h5",save_best_only=True)]

    epochs = 100
    model.fit(train_gen,epochs=epochs,validation_data=train_gen,callbacks = callbacks)

def model_pred():
    model_center = load_model("ctc_train/center_cell.h5",compile=False)
    for img_file in os.listdir('/home/tom/ctc_train/Fluo-C3DL-MDA231/02'):
        img = imread(os.path.join('/home/tom/ctc_train/Fluo-C3DL-MDA231/02',img_file))
        img_new = np.zeros((32,512,512))
        img_new[0:30]=img
        img_inputs = np.expand_dims(img_new,3)
        img_inputs = np.expand_dims(img_inputs,0)
        output = model_center.predict(img_inputs)
        imsave(os.path.join('/home/tom/ctc_train/MDA231_test/',img_file),output)

#train()
model_pred()
