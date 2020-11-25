# Emotion-Detection-for-Tom-and-Jerry
Given a set of images of our favourite cartoon Tom and Jerry with some labelled emotion images, given the test files classify them on the basis of 4 major emotions: angry , happy,sad ,surprised and one unknown if it doesnot classify as the other four classes.
# Model Used: Convolutional Neural Networks

Reason: CNN was used to detect the particular features which relate to the
appropriate emotion detection parameter.Each CNN contains a
convolutional layer and a max-pooling layer, followed by a fully-
connected layer for classifying the images.CNN helps us to detect the
particular emotion even if it is at different locations in different images.
Also , the major reason I chose CNN was on the training set it give more
accuracy than MLP ,SVM and Random Forest.

# Problems Faced:
The training data that was given to us was less in size and we needed more
versatile data and more voluminous data to train our emotion detection as
more data means a better model.

# Solution to the Problem:
We can translate our images and introduce random parameters which will
modify the images in sense such that for eg the original image is rotated ,
flipped etc and thus the size of the training data will increase and it can be
used for predicting the test images and for building a better model.
