#evaluate
#--------------0- Load requirements----------------------------------
import imageio
from skimage import transform,io
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
size=150
model_path='./data/my_model.h5'
model = tf.keras.models.load_model(model_path)
print("model loaded")

#either add pictures to intriguing pics or change path to the dir containing the pics you want to test
#In Jupyter you ll have the plots as well and its much better
#--------------1- Test on all images on path----------------------------------
path="./data/Testing intriguing pics"
for x in os.listdir(path):
    imgarray = imageio.imread(os.path.join(path,x), pilmode='RGB')
    imgresized = transform.resize(imgarray, (size,size), mode='symmetric')
    result=model.predict(imgresized.reshape(-1,size,size,3))
    winner=np.max(result)
    ele=result[0]
    print(x)
    print(ele)
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
    plt.imshow(imgresized, cmap="gray")
    plt.show()