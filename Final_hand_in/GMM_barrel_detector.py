'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os, cv2
from skimage.measure import label, regionprops
import numpy as np
import cv2
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.morphology import label, closing, square
from skimage.measure import regionprops

class BarrelDetector():
	def __init__(self):
		'''
			Initilize your blue barrel detector with the attributes you need
			eg. parameters of your classifier
		'''

		self.alpha = np.load('alpha_K4.npy')
		self.sigma = np.load('sigma_K4.npy')
		self.mu = np.load('mu_K4.npy')
		"""
		self.alpha =np.array([[0.30649246],
				[0.08440951],
				[0.60909803]]) 
		self.sigma = np.array([[[  37.20590416,   -0.    ,        0.        ],
					[  -0.   ,       261.41350613  ,  0.        ],
  					[   0.      ,      0.     ,     431.67385095]],

 					[[9614.40565013   , 0.    ,       -0.        ],
					[   0.      ,   2007.96942345   ,-0.        ],
					[  -0.        ,   -0.    ,     1792.35814471]],

					[[  43.20526411  ,  0.       ,     0.        ],
					[   0.      ,   1313.65564059  , -0.        ],
					[   0.      ,     -0.      ,   1945.11545337]]]) 
		self.mu = np.array([[ 29.49395146, 235.11014787, 182.6499033 ],
							[108.47016151 , 94.69981127 , 49.04835968],
							 [ 21.47508687, 188.32532304, 114.21469647]]) 
		"""					
		#raise NotImplementedError
	def pdf(self,x, sigma, mu):
		"""

		:param x: n*3
		:param sigma: 3*3 diag
		:param mu: 1*3 (for certain k)
		:return: prob 1*n
		"""
		n, dim = np.shape(x)
		A = np.linalg.inv(sigma)
		# prod = A[0, 0] * (diff[:, 0] ** 2 + diff[:, 1] ** 2 + diff[:, 2] ** 2)
		diff = x - mu
		prod = np.multiply(np.dot(diff, A), diff)
		prod = np.sum(prod, axis = 1)
		# prod=np.dot(diff.transpose(),sigma)
		# prod=np.dot(prod, diff)
		prob = ( 1/np.linalg.det(sigma) / ((2 * np.pi) ** dim))**0.5 * np.exp(-0.5 * prod)

		return prob
	def pred(self,img, mu, sigma, alpha):
		"""

		:param img: h*w*3
		:param mu: k*3
		:param sigma: k*3*3 diag
		:param alpha: k*1
		:return: p h*w
		"""
		img = img[0]
		h, w, z = np.shape(img)
		n = h * w
		data = np.zeros([n, 3])
		data[:, 0] = img[:, :, 0].flatten()
		data[:, 1] = img[:, :, 1].flatten()
		data[:, 2] = img[:, :, 2].flatten()
		k, dim = np.shape(mu)

		alphamat=np.tile(alpha.transpose(), (n,1))
		px = np.zeros([k, n])
		for j in range(0, k):
			px[j,:] = self.pdf(data, sigma[j], mu[j, :])
		px = px.transpose()
		p=px*alphamat
		p=np.sum(p, axis=1)
		p = np.reshape(np.log(p), [h, w])
		return p
	
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
		#raise NotImplementedError
		#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		imgs_hsv=[]
		pic = img
		pic = pic /(255 - 0) *(1 - 0)+0
		pic = pic / (1 - 0) * (255 - 0) + 0
		pic = pic.astype(np.float32)
		#pic = np.full((800,1200,3), 12, np.uint8)
		pic = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)#2YCR_CB)
		pic[:, :, 1] = pic[:, :, 1] / (1 - 0) * (255 - 0) + 0
		imgs_hsv.append(pic)
		#print(imgs_hsv,np.shape(imgs_hsv))#,np.shape(imgs_hsv[1]))
		p = self.pred(imgs_hsv, self.mu, self.sigma, self.alpha)
		mask_img=(p > max(-16.99, np.percentile(p, 98)))#(p>-15)
		return mask_img
	def refine_mask(self,mask,img, n):
		"""

		:param mask: h*w mask of red labels
		:param n: scalar for dilation iteration
		:return: detected h*w refined mask
		"""
		mask = mask.astype(np.uint8)
		#blurred=cv2.medianBlur(mask, ksize=3)
		kernel = np.ones((2, 2), np.uint8)
		erosed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations = 2)
		closing= cv2.morphologyEx(erosed, cv2.MORPH_CLOSE, kernel)
		detected = cv2.dilate(closing, kernel, iterations = n)
		
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
			
		return detected
	def get_bounding_box(self, img):
		'''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom righ  t coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		# YOUR CODE HERE
		#raise NotImplementedError
		boxes=[]
		#obbox = []

		mask_img = self.segment_image(img)
		re_mask = self.refine_mask(mask_img,img,8)
		mask_blue = label(re_mask)
		#mask_blue = label(mask_img)
		for region in regionprops(mask_blue):
			#ominr, ominc, omaxr, omaxc=region.bbox
			#obbox.append(np.asarray([ominr, ominc, omaxr, omaxc]))
			if region.area < 1000:
				continue
			minr, minc, maxr, maxc = region.bbox
			h=abs(maxr-minr)
			w=abs(maxc-minc)
			if 1.05 < h / w < 2.2:
				if (region.area /(w*h))<0.3 :
					continue
				x1 = minc
				y1 = minr
				x2 = maxc
				y2 = maxr
				boxes.append(np.asarray([x1, y1, x2, y2])) #boxes.append(np.asarray([minr, minc, maxr, maxc]))
		
		print(boxes)
		return boxes


if __name__ == '__main__':
	#Folder ="Train_Validation_set/"
	imgFolder = "validationset/"
	my_detector = BarrelDetector()
	for filename in os.listdir(imgFolder):
		# read one test image
		Imagename, extension = os.path.splitext(filename)
		img = cv2.imread(os.path.join(imgFolder,filename))
		#img = plt.imread(imgFolder + filename)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		mask_img = my_detector.segment_image(img)
		
		"""
		fig = plt.figure()
		ax1 = fig.add_subplot(121)
		ax1.imshow(img)
		ax2 = fig.add_subplot(122)
		ax2.imshow(mask_img)
		plt.show()
		"""
		img_cp=img.copy()
		boxes = my_detector.get_bounding_box(img)
		#print('box is :',boxes)
		#img_cp = cv2.cvtColor(img_cp, cv2.COLOR_RGB2BGR)
		for j in range(len(boxes)):
			sides=boxes[j]
			img_cp = cv2.rectangle(img_cp, (sides[0], sides[1]), (sides[2], sides[3]), (0, 255, 0), 2)
		#for j in range(len(obbox)):
		#	sides1=obbox[j]
		#	img_cp = cv2.rectangle(img_cp, (sides1[1], sides1[0]), (sides1[3], sides1[2]), (255, 0, 0), 2)
		#fig = plt.figure()
		'''
		fig = plt.figure()
		ax1 = fig.add_subplot(221)
		ax1.imshow(img)
		ax4 = fig.add_subplot(222)
		#ax4.imshow(erosed)
		ax4.imshow(mask_img)
		ax2 = fig.add_subplot(224)
		ax2.imshow(img_cp)
		#ax2.imshow()
		#ax3 = fig.add_subplot(224)
		#ax3.imshow(detected)
		plt.show()
		'''
		#cv2.imshow('image', img_cp)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		#img = []
		#img = cv2.imread(os.path.join(folder,filename))
		#cv2.imshow('image', img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		#Display results:
		#(1) Segmented images
		#	 mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box
		#    boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope

