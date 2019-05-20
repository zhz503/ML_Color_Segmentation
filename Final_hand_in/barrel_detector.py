# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 23:55:14 2019

@author: Zhaoliang
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from skimage.measure import label, regionprops
from PIL import Image
import skimage.io
from LR_TrainModel_good import sigmoid

class BarrelDetector():
    def __init__(self):
        '''
			Initilize your blue barrel detector with the attributes you need
			eg. parameters of your classifier
		'''
        #self.weights =  np.array([-736.5 ,-721.5, 1055.5 ])
        #self.weights =  np.array([ -139.34, -78.2, 115.254])
        self.weights =  np.array([ -57.01 ,-28.50 ,43.03]) # good result
        #self.weights =  np.array([ -18914.5 , -46041.08, 47646.87]) # GOOD RESULT
        #self.weights =  np.array([ -18914.5 , -46041.08, 40646.87])#self.weights = np.load('parameters.npy')
        #raise NotImplementedError
    def refine_mask(self,mask,img, n):
        """

		:param mask: h*w mask of red labels
		:param n: scalar for dilation iteration
		:return: detected h*w refined mask
        """
        mask = mask.astype(np.uint8)
        kernel = np.ones((2, 2), np.uint8)
        erosed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
        closing= cv2.morphologyEx(erosed, cv2.MORPH_CLOSE, kernel)
        detected = cv2.dilate(closing, kernel, iterations = n)
        """
		fig = plt.figure()
		ax1 = fig.add_subplot(221)
		#ax1.imshow(mask)
		ax1.imshow(img)
		ax4 = fig.add_subplot(222)
		#ax4.imshow(erosed)
		ax4.imshow(mask)
		ax2 = fig.add_subplot(223)
		ax2.imshow(closing)
		#ax2.imshow()
		ax3 = fig.add_subplot(224)
		ax3.imshow(detected)
		plt.show()
        """
        return detected
    def segment_image(self, img):
        '''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		# YOUR CODE HERE
        rgbimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h, w, z = np.shape(rgbimg)
        mask =  np.zeros((h, w))
        for i in range(h):
           for j in range(w):
               binary = sigmoid(np.dot(self.weights,rgbimg[i,j,:]))
               mask[i,j] = binary
        mask_img = np.asarray(mask)
        #raise NotImplementedError
        return mask_img

        
    def get_bounding_box(self, img):
        '''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
        		# YOUR CODE HERE
        #raise NotImplementedError
        boxes=[]
        mask_img = self.segment_image(img)

        re_mask = self.refine_mask(mask_img,img,8)
        mask_blue = label(re_mask)
        for region in regionprops(mask_blue):
            if region.area < 1000:
                continue
            minr,minc, maxr, maxc = region.bbox
            h=abs(maxr-minr)
            w=abs(maxc-minc)
            if 1.05 < h / w < 2.5:
                if (region.area /(w*h))<0.3 :
                    continue
                x1 = minc
                y1 = minr
                x2 = maxc
                y2 = maxr
                boxes.append(np.asarray([x1, y1, x2, y2])) #boxes.append(np.asarray([minr, minc, maxr, maxc]))
		
		#print(bbox)

        return boxes


if __name__ == '__main__':
    imgFolder = "validationset/"
    my_detector = BarrelDetector()
    for filename in os.listdir(imgFolder):
        # read one test image
        Imagename, extension = os.path.splitext(filename)
        img = cv2.imread(os.path.join(imgFolder,filename))
        
        
        mask_img = my_detector.segment_image(img)
        rgbimg= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_cp = img.copy()
        boxes = my_detector.get_bounding_box(img)
        #x1,y1,x2,y2 =boxes
        for j in range(len(boxes)):
            sides=boxes[j]
            img_cp = cv2.rectangle(img_cp, (sides[0], sides[1]), (sides[2], sides[3]), (0, 255, 0), 2)
        #boxes = my_detector.get_bounding_box(img)
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax1.imshow(rgbimg)
        ax4 = fig.add_subplot(122)
        ax4.imshow(mask_img)
        plt.show()
        cv2.imshow('image', img_cp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(boxes)
