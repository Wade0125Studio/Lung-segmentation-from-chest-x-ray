import numpy as np 
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
import pandas as pd
from tqdm import tqdm
import os
from cv2 import imread, createCLAHE 
import cv2
from glob import glob
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow.keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
from plot_keras_history import show_history, plot_history
from tensorflow.keras import layers



image_path = os.path.join("C:/Users/GIGABYTE/Downloads/lung-segmentation-from-chest-x-ray-dataset/Lung Segmentation/CXR_png/")
mask_path = os.path.join("C:/Users/GIGABYTE/Downloads/lung-segmentation-from-chest-x-ray-dataset/Lung Segmentation/masks/")
images = os.listdir(image_path)
mask = os.listdir(mask_path)
mask = [fName.split(".png")[0] for fName in mask] 
image_file_name = [fName.split("_mask")[0] for fName in mask] 
check = [i for i in mask if "mask" in i]
print("Total mask that has modified name:",len(check))
testing_files = set(os.listdir(image_path)) & set(os.listdir(mask_path))
training_files = check




def getData(X_shape, flag = "test"):
    im_array = []
    mask_array = []
    if flag == "test":
        for i in tqdm(testing_files): 
            im = cv2.resize(cv2.imread(os.path.join(image_path,i)),(X_shape,X_shape))[:,:,0]
            mask = cv2.resize(cv2.imread(os.path.join(mask_path,i)),(X_shape,X_shape))[:,:,0]
            im_array.append(im)
            mask_array.append(mask)
        return im_array,mask_array
    if flag == "train":
        for i in tqdm(training_files): 
            im = cv2.resize(cv2.imread(os.path.join(image_path,i.split("_mask")[0]+".png")),(X_shape,X_shape))[:,:,0]
            mask = cv2.resize(cv2.imread(os.path.join(mask_path,i+".png")),(X_shape,X_shape))[:,:,0]
            im_array.append(im)
            mask_array.append(mask)
        return im_array,mask_array


def plotMask(X,y):
    sample = []
    for i in range(6):
        left = X[i]
        right = y[i]
        combined = np.hstack((left,right))
        sample.append(combined)
    for i in range(0,6,3):
        plt.figure(figsize=(25,10))
        plt.subplot(2,3,1+i)
        plt.imshow(sample[i],cmap='gray')
        plt.subplot(2,3,2+i)
        plt.imshow(sample[i+1],cmap='gray')
        plt.subplot(2,3,3+i)
        plt.imshow(sample[i+2],cmap='gray')
        
        plt.show()



dim = 256
X_train,y_train = getData(dim,flag="train")
X_test, y_test = getData(dim)
print("training set")
plotMask(X_train,y_train)



X_train = np.array(X_train).reshape(len(X_train),dim,dim,1)
y_train = np.array(y_train).reshape(len(y_train),dim,dim,1)
X_test = np.array(X_test).reshape(len(X_test),dim,dim,1)
y_test = np.array(y_test).reshape(len(y_test),dim,dim,1)
assert X_train.shape == y_train.shape
assert X_test.shape == y_test.shape
images = np.concatenate((X_train,X_test),axis=0)
mask  = np.concatenate((y_train,y_test),axis=0)




def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
def SegNet(input_shape, num_classes):

    inputs = layers.Input(shape=input_shape)

    # encoder
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x= layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x= layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # decoder
    x = UpSampling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = UpSampling2D((2, 2))(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = UpSampling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)

    x = UpSampling2D((2, 2))(x)
    x = layers.Convolution2D(128, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    x = layers.Convolution2D(64, (3, 3), padding='same', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_classes, (1, 1), padding='valid', kernel_initializer='he_uniform')(x)
    # outputs = layers.BatchNormalization()(x)
    outputs = layers.Activation('sigmoid')(x)

    segnet_model = Model(inputs=inputs, outputs=outputs, name='SegNet')
    segnet_model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss,metrics=[dice_coef, 'binary_accuracy'])
    return segnet_model

model = SegNet(input_shape=(256,256,1),num_classes=1)
#model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss,metrics=[dice_coef, 'binary_accuracy'])           
model.summary()



from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
weight_path="C:/Users/GIGABYTE/Downloads/lung-segmentation-from-chest-x-ray-dataset/{}_weights.best.hdf5".format('cxr_reg_Segnet')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,save_best_only=True, mode='min') 
                            
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                   patience=3, 
                                   verbose=1, mode='min', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_loss",mode="min", patience=15) 
                      

callbacks_list = [checkpoint, early, reduceLROnPlat]



train_vol, validation_vol, train_seg, validation_seg = train_test_split((images-127.0)/127.0, 
                                                            (mask>127).astype(np.float32), 
                                                            test_size = 0.1,random_state = 2018)

train_vol, test_vol, train_seg, test_seg = train_test_split(train_vol,train_seg, 
                                                            test_size = 0.1, 
                                                            random_state = 2018)

loss_history = model.fit(x = train_vol,y = train_seg,batch_size = 8,
                  epochs = 50,validation_data =(test_vol,test_seg) ,
                  callbacks=callbacks_list)

model.save('C:/Users/GIGABYTE/Downloads/lung-segmentation-from-chest-x-ray-dataset/SEGNET_SEG_model.h5') 


show_history(loss_history)
plot_history(loss_history, path="C:/Users/GIGABYTE/Downloads/lung-segmentation-from-chest-x-ray-dataset/Training_history SEGNET.png",
             title="Training history SEGNET")



pred_candidates = np.random.randint(1,validation_vol.shape[0],10)
preds = model.predict(validation_vol)
plt.figure(figsize=(20,10))
for i in range(0,9,3):
    plt.subplot(3,3,i+1)
    plt.imshow(np.squeeze(validation_vol[pred_candidates[i]]),cmap='gray')
    plt.title("Base Image")
    plt.subplot(3,3,i+2)
    plt.imshow(np.squeeze(validation_seg[pred_candidates[i]]),cmap='gray')
    plt.title("Mask")
    plt.subplot(3,3,i+3)
    plt.imshow(np.squeeze(preds[pred_candidates[i]]),cmap='gray')
    plt.title("Pridiction")
    plt.suptitle("Segnet predict result")
plt.savefig('C:/Users/GIGABYTE/Downloads/lung-segmentation-from-chest-x-ray-dataset/SEGNET_predict.png')    




