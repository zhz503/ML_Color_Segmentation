# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 21:05:21 2019

@author: Zhaoliang
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from roipoly import roipoly
from skimage.measure import label, regionprops
from PIL import Image
import skimage.io

#################### Variables name ####################
Bluepixels = []
imgs = []
bpixels_Valid = []
bpixels_Train = []
imgs_Valid = []
imgs_Train = []
#################### Folder name ####################
Folder = "labeled_img/Barrel_Blue"
imgFolder = "trainset/"
TVFolder = "Train_Validation_set/"
#################### Load npy data ####################
for filename in os.listdir(imgFolder):
    #print(filename)
    Imagename, extension = os.path.splitext(filename)
    pics = plt.imread(imgFolder + filename)
    #bgrImage = cv2.imread(imgFolder + filename)
    #rgbImage = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)#plt.imread
    
    npyname = Imagename + ".npy"
    blue_p = np.load(os.path.join(Folder, npyname))

    imgs.append(pics)
    Bluepixels.append(blue_p)

#################### Train_Validation_Split ####################
ToValid = np.array([36,37,38,3940,41,42,43,44,45,46])#([42,46,17,15,5])#([0,1,2,3,4,5,6,7,8])##
n=len(Bluepixels)
for i in  range(0, n):
    if any(ToValid == i):
        bpixels_Valid.append(Bluepixels[i])
        imgs_Valid.append(imgs[i])

    else:
        bpixels_Train.append(Bluepixels[i])
        imgs_Train.append(imgs[i])

#################### Save data to npy ####################
np.save(TVFolder + 'bpixels_Train.npy',bpixels_Train)
np.save(TVFolder + 'imgs_Train.npy',imgs_Train)
np.save(TVFolder + 'bpixels_Valid.npy',bpixels_Valid)
np.save(TVFolder + 'imgs_Valid.npy',imgs_Valid)
#################### Variables name ####################
Bluepixels_mask = []
imgs = []
HSVData = []
rgbData  = np.zeros([3, 1])
#################### Folder name ####################
Bluepixels_mask = bpixels_Train
imgs =  imgs_Train
n = len(imgs)
for i in range(n):
	rgbImg = imgs[i]
	imgs_HSV = rgbImg
	b_mask = Bluepixels_mask[i]
	rgbImg = rgbImg * 255 
    #To RGB
	r, g, b = cv2.split(rgbImg)
	red = np.array([r[b_mask]])
	green = np.array([g[b_mask]])
	blue = np.array([g[b_mask]])
    
	#rg = np.append(red.transpose(), green.transpose(), axis = 1)
	#rgb = np.append(rg, blue.transpose(), axis = 1)
	#rgbData = np.append(rgbData, rgb, axis = 0)

	rg = np.append(red, green ,axis = 0)
	rgb = np.append(rg, blue ,axis = 0)
	#plt.imshow(rgb)
	#plt.show()
	rgbData = np.append(rgbData, rgb, axis = 1)
    #to HSV
	imgs_HSV = cv2.cvtColor(imgs_HSV, cv2.COLOR_RGB2HSV)
	H = imgs_HSV[:,:,0]
	s =imgs_HSV[:,:,1]
	v = imgs_HSV[:,:,2]
	HSVData.append(imgs_HSV)

np.save(TVFolder +'rgbData.npy',rgbData)
