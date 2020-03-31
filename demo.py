#--------------0- Load requirements----------------------------------
import imageio
from skimage import transform,io
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

model_path='./data/my_model.h5'
img_path='./data/Testing intriguing pics/harvardstanford.jpg'
model = tf.keras.models.load_model(model_path)
print("model loaded")
size=150

#--------------1- Test on img----------------------------------
imgarray = imageio.imread(img_path, pilmode='RGB')
imgresized = transform.resize(imgarray, (size,size), mode='symmetric')
result=model.predict(imgresized.reshape(-1,size,size,3))
winner=np.max(result)
print(img_path)
for ele in result:
    print(result)
    if ele[0]>=winner:
        print("MIT !!!!!!!!!!")
    if ele[1]>=winner:
        print("Arts et métiers Paritech !!!!")
    if ele[2]>=winner:
        print("X !!!!!!!!!!!")
    if ele[3]>=winner:
        print("STANFORD!!!!!!!!")
    if ele[4]>=winner:
        print("CentraleSupélec!!!!!!!!")
    if ele[5]>=winner:
        print("HARVARD ITCH!!!!!!!!")
print("The winner got a score of "+ str(np.max(result)))
plt.imshow(imgarray)
plt.show()