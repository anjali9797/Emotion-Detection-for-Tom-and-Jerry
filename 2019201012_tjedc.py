#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.image as img
import matplotlib.pyplot as plt
from keras.preprocessing import image
from PIL import Image


# In[2]:


#loading the data with its labels
df=pd.read_csv("img_label.csv")
#separating img names and their corresponding labels
img_names=df.get("image_file")
labels=df.get("emotion")
img_names=list(img_names)
labels=list(labels)
#reading all the images:
loc_img=list()
for i in img_names:
    path="/home/anjali/smai4/train_images/"+i+".jpg"
    loc_img.append(path)
imgs = []
w=60
h=60
c=0
labbs=list()
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=2,
        width_shift_range=0.002,
        height_shift_range=0.002,
        shear_range=0.002,
        zoom_range=0.002,
        horizontal_flip=True,
        fill_mode='nearest')
for f in loc_img:
    img =tf.keras.preprocessing.image.load_img(f)
    img = img.resize((w,h), Image.ANTIALIAS)
    img=tf.keras.preprocessing.image.img_to_array(img)/255
    img = img.reshape( w,h,3)
    imgs.append(img)
    labbs.append(labels[c])
    #generating more images to increase the size of our data set and so that training can be carried out in an efficient manner
    # this is a Numpy array with shape (3, 150, 150)
    x=img.reshape((1,) + img.shape) 
    i = 0
    for batch in datagen.flow(x, batch_size=1):
        i=i+1
        bat=batch.reshape(60,60,3)
        imgs.append(bat)
        labbs.append(labels[c])
        if i>5:
            break
    imgs.append(img)
    labbs.append(labels[c])
    imgs.append(img)
    labbs.append(labels[c])
    imgs.append(img)
    labbs.append(labels[c])
    c=c+1
    
imgs=np.array(imgs)



# In[3]:

"""
#splitting in train and test split
ls=list()
lab=list()
test_data=list()
actual_labels=list()
labels=labbs
print(len(imgs))
#splitting part

for x in range(len(imgs)-1):
    if np.random.random()<0.77:
        ls.append(imgs[x])
        lab.append(labbs[x])
    else:
        test_data.append(imgs[x])
        actual_labels.append(labbs[x])

"""
# In[4]:


x_train=imgs
y_train=labbs


# In[6]:


#defining a cnn model
model_cnn = tf.keras.models.Sequential()
model_cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same",input_shape=(60,60,3)))
#model_cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding="same"))
model_cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
model_cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',padding="same"))
model_cnn.add(tf.keras.layers.MaxPooling2D((2, 2)))
model_cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',padding="same"))
model_cnn.add(tf.keras.layers.MaxPooling2D((9, 9)))
model_cnn.add(tf.keras.layers.Flatten())
model_cnn.add(tf.keras.layers.Dense(512, activation='relu'))
model_cnn.add(tf.keras.layers.Dense(5, activation='softmax'))
 
model_cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
lenn=len(x_train)
#print(lenn)
x_train=np.array(x_train)
x_train = x_train.reshape((lenn, 60, 60, 3))

#x_test=np.array(x_test)
#x_test = x_test.reshape((len(x_test) ,60, 60, 3))
model_cnn.fit(x_train, y_train, epochs=55)


# In[ ]:


"""
#validating part
x_test=np.array(test_data)
y_test=actual_labels
print(x_train.shape)
print(x_test.shape)
val_loss, val_acc = model_cnn.evaluate(x_test, y_test)
print("Loss:",val_loss)
print("Accuracy:",val_acc)
y_pred=model_cnn.predict_classes(x_test)
from sklearn.metrics import f1_score
print("F1 Score:",f1_score(y_pred,y_test,average='weighted'))
"""


# In[7]:


#reading the test data for predicting the values for it
#loading the data with its labels
df=pd.read_csv("img_test.csv")
#separating img names and their corresponding labels
img_na=df.get("image_file")

img_na=list(img_na)
#print(img_na)
#reading all the images:
loc_img=list()
for i in img_na:
    path="/home/anjali/smai4/test_images/"+i+".jpg"
    loc_img.append(path)
imgs = []
w=60
h=60
c=0

for f in loc_img:
    img =tf.keras.preprocessing.image.load_img(f)
    img = img.resize((w,h), Image.ANTIALIAS)
    img=tf.keras.preprocessing.image.img_to_array(img)/255
    img = img.reshape( w,h,3)
    imgs.append(img)
    
x_test=np.array(imgs)
y_pred=model_cnn.predict_classes(x_test)


# In[19]:





# In[9]:


f1=open("submission2.csv","w")
f1.write("emotion")
for i in y_pred:
    st=str(i)
    f1.write("\n")
    f1.write(st)
f1.close()


# In[13]:






# In[ ]:




