# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 14:44:09 2020

@author: zyclyx
"""

import cv2
import train as osl
import numpy as np
import os



model_path = './weights/'
osl.model.load_weights(os.path.join(model_path, "weights.13400.h5"))


def img_to_encoding(image,image2):
    image = cv2.resize(image, (105, 105)) 
    img = image[...,np.newaxis]
    image2 = cv2.resize(image2, (105, 105)) 
    img2 = image2[...,np.newaxis]
    x_train = np.array([img])
    x_train2 = np.array([img2])
    embedding = osl.model.predict([x_train,x_train2])
    if(embedding[0][0] > 0.65):
        print("match :", embedding[0][0])
        
        res=1
    else:
        print("Not match :", embedding[0][0])
        
        res=0
    return res



im1=cv2.imread(r"/home/zyclyx/Documents/sprint-15/signature/signature/testpairs/kri.png",0)
im2=cv2.imread(r"/home/zyclyx/Documents/sprint-15/signature/signature/testpairs/kri2.png",0)


res=img_to_encoding(im1,im2)
print('result : ',res)
