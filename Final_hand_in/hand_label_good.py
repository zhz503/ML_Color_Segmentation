# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:57:12 2019

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

def onKey(event, labelImgname, rgbImg):
	global redo
	global rois
	if event.key == 'a':
		redo = True
		plt.close()
	elif event.key == 'r':
		rois = []
		redo = True
		plt.close()
	elif event.key == 'd':
		redo = False
		imgShape = np.shape(rgbImg)
		img_B =rgbImg[:,:,2]
		img_mask = np.zeros(imgShape[0:2], dtype=bool)
		for roi in rois:
			img_mask = img_mask + roi.get_mask(img_B) 
		np.save(labelImgname, img_mask)
		plt.close()

if __name__ == '__main__':
	imgFolder = "trainset/"
	labeledFolder = "labeled_img/"
	colorFolder2 = "Barrel_Blue/"	
	colorFolder1 = "LR_Blue/"
	for filename in os.listdir(imgFolder):
		Imagename, extension = os.path.splitext(filename)
		outPath = labeledFolder + colorFolder1
		redo = True
		rois = []
		labelImgname = outPath + Imagename + ".npy"
		if os.path.isfile(labelImgname):
			continue
		bgrImage = cv2.imread(imgFolder + filename)
		rgbImg = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2RGB)
		while (redo):
			fig =plt.figure()
			plt.imshow(rgbImg)
			for roi in rois:
				roi.displayROI()
			plt.title(Imagename + ".png")
			rois.append(roipoly(roicolor='r'))
			fig = plt.gcf()
			fig.canvas.mpl_connect('key_press_event',lambda event: onKey(event, labelImgname, rgbImg))
			plt.imshow(rgbImg)
			plt.title("Press \'d\' to save and move on \n \'r\' to redo|\'a\' to add another region")
			for roi in rois:
				roi.displayROI()
			plt.show()



