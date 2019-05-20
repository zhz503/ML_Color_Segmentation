# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:05:21 2019

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

def EM_init(data,k):
    sigma = []
    alpha = np.ones([k,1])/k
    (n,dim) = data.shape
    range1 = np.amax(data[:,0])-np.amin(data[:,0])
    range2 = np.amax(data[:,1])-np.amin(data[:,1])
    range3 = np.amax(data[:,2])-np.amin(data[:,2])
    for i in range(k):
        sigmai = np.array([[range1/2,0,0],[0,range2/2,0],[0,0,range3/2]])
        sigma.append(sigmai)
    #seed = np.random.choice(n, k, replace = False)
    #mu = np.reshape(data[seed,:],[k,3])
    mu = data[np.random.choice(n, k, replace = False), :]
    mu = np.reshape(mu, [k, 3])
    return alpha, mu, sigma

def guassPDF(data,mu,sigma):
    (n,dim) = data.shape
    data = data - mu
    prob = np.sum(np.multiply(np.dot(data,np.linalg.inv(sigma)),data),1)
    prob = np.exp(-0.5*prob)/np.sqrt( 1/np.linalg.det(sigma) / ((2 * np.pi) ** dim))
    return prob

def predict(img, mu, sigma, alpha):
	h, w, z = np.shape(img)
	n = h * w
	data = np.zeros([n, 3])
	data[:, 0] = img[:, :, 0].flatten()
	data[:, 1] = img[:, :, 1].flatten()
	data[:, 2] = img[:, :, 2].flatten()
	k, dim = np.shape(mu)

	alphamat = np.tile(alpha.transpose(), (n, 1))
	px = np.zeros([k, n])
	for j in range(0, k):
		px[j, :] = guassPDF(data, sigma[j], mu[j, :])
	px = px.transpose()

	p = px * alphamat
	p = np.sum(p, axis = 1)
	p = np.reshape(np.log(p), [h, w])

	return p