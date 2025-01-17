import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import matplotlib.patches as patches
import tensorflow as tf
import PIL
from PIL import Image
import random
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import imageio
from skimage import transform,io
from distutils.dir_util import copy_tree
import shutil
import os
import sys
#--------------0- Init-----------------------------------------------------
#Feel free to add more images of logos in ./data/photodata
#Deleting augmentated images from last time
print ('Argument List:', str(sys.argv[-1]))
shutil.rmtree("./data/augdata/images")
#Copy all images we have from the ressource file ./data/photodata
shutil.copytree("./data/photodata/", "./data/augdata/images/")
print("Folder Copied !")
#Parameters
categories=["MIT","AM","X","STAN","CS","HARV"]
folder="./data/augdata/images"
size=64
data=[]
aug=10
print("Ready to roll !")
#--------------1- Lets build our Augmentated dataset in augdata-------------
#vertical_flip=True
#horizontal_flip=True,
datagen = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        brightness_range=[0.2,1.0])
for cat in categories:
    data=[]
    pathdata=os.path.join(folder,cat)
    class_num=categories.index(cat)
    for img in os.listdir(pathdata):        
        imgarray=Image.open(os.path.join(pathdata,img)).convert('RGB')
        imgresized = imgarray.resize((size,size)) 
        #plt.imshow(imgresized)
        #plt.show()
        data.append([np.array(imgresized),class_num])
    #Augmentate Data for each cat
    random.shuffle(data)
    X=[]
    y=[]
    for features, label in data:
        X.append(features)
        y.append(label)
    X=np.array(X).reshape(-1,size,size,3)
    y=np.array(y)
    #fit
    datagen.fit(X)
    batches=datagen.flow(X, y,save_to_dir=pathdata,
                  save_prefix=cat, save_format='jpg')
    for i in range(aug):
        next(batches)
        
#------------------------ 2- Lets build our Training and Testing Set-----------------------
categories=["MIT","AM","X","STAN","CS","HARV"]
#if you want augmented dataset uncomment next line
folder="./data/augdata/images"
#folder=".\photodata"
data=[]
for cat in categories:
    pathdata=os.path.join(folder,cat)
    class_num=categories.index(cat)
    for img in os.listdir(pathdata):
        imgarray=Image.open(os.path.join(pathdata,img)).convert('RGB')
        imgresized = imgarray.resize((size,size)) 
        #plt.imshow(imgresized)
        #plt.show()
        data.append([np.array(imgresized),class_num])
import random
random.shuffle(data)
X=[]
y=[]
for features, label in data:
    X.append(features)
    y.append(label)
    
X=np.array(X).reshape(-1,size,size,3)
y=pd.DataFrame(y)
#from sklearn.preprocessing import OneHotEncoder
#enc = OneHotEncoder()
#y=enc.fit_transform(y.reshape(-1,1)).toarray()
y=pd.get_dummies(y.astype(str),prefix=['label'])

print("The dataset size is")
print(X.shape,y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
print("The trainingset size is")
print(X_train.shape,y_train.shape)
print("The testingset size is")
print(X_test.shape,y_test.shape)


#------------------------ 3- Let's Build THE CNN -----------------------
droprate=0.2
#Architecture
c1,c2,c3,c4,d1,d2,d3=32,32,64,64,128,256,256
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(c1,(3,3),activation="relu",input_shape=(size,size,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(c2,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(c3,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(c4,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(droprate),
    tf.keras.layers.Dense(d1,activation="relu"),
    tf.keras.layers.Dropout(droprate),
    tf.keras.layers.Dense(d2,activation="relu"),
    tf.keras.layers.Dropout(droprate),
    tf.keras.layers.Dense(d3,activation="relu"),
    tf.keras.layers.Dense(6,activation="softmax")
])
#opt=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

#------------------------ 4- Training-----------------------
from datetime import datetime
#datetime.now().strftime("%Y%m%d-%H%M%S")
ext="_"+str(size)+"_"+str(aug)+"_"+str(droprate)+"_"+str(c1)+"x"+str(c2)+"x"+str(c3)+"x"+str(c4)+"x"+str(d1)+"x"+str(d2)+"x"+str(d3)
logdir = "./logs/scalars/model"+ext
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(
    X_train, # input
    y_train, # output
    verbose=1, # Suppress chatty output; use Tensorboard instead
    epochs=50,
    validation_data=(X_test,y_test),
    callbacks=[tensorboard_callback],
)

#------------------------ 6- Save/Load Model-----------------------
model.save('./data/models/model'+ext+'.h5')  # creates a HDF5 file 'my_model.h5'
print("model saved")
#model = tf.keras.models.load_model('./data/my_modelx.h5')
#print("model loaded")
