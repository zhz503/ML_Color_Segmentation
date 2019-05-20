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
from EM_init_good import EM_init,guassPDF,predict

def Estep(x, mu, sigma, alpha):

	n, dim = np.shape(x)
	k, dim = np.shape(mu)
	r = np.zeros([n, k])

	for z in range(0, k):
		r[:, z] = alpha[z, :] * guassPDF(x, sigma[z], mu[z, :])
	# pdb.set_trace()
	r = np.divide(r, np.tile(np.sum(r, axis = 1), (k, 1)).transpose())
	return r

def Mstep(x, r):

	n, k = np.shape(r)
	sigma = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])] * k
	mu = np.zeros([k, 3])
	alpha = np.zeros([k, 1])
	for i in range(0, k):
		alpha[i, :] = 1 / n * sum(r[:, i:i + 1])
		mu[i, 0] = sum(r[:, i:i + 1] * x[:, 0:1]) / sum(r[:, i:i + 1])
		mu[i, 1] = sum(r[:, i:i + 1] * x[:, 1:2]) / sum(r[:, i:i + 1])
		mu[i, 2] = sum(r[:, i:i + 1] * x[:, 2:3]) / sum(r[:, i:i + 1])

		diff = x - mu[i:i + 1, :]

		rcut = r[:, i:i + 1]
		rcut = rcut[:, :, np.newaxis]
		pvalue = np.ones([n, 3, 3])
		pvalue = rcut * pvalue  # n*3*3

		span = diff[:, :, np.newaxis]
		diffmat = np.ones([n, 3, 3])
		diffmat = span * diffmat  # n*3*3
		diffmat_t = np.transpose(diffmat, (0, 2, 1))  # n*3*3

		sigmak = np.multiply(pvalue, np.multiply(diffmat, diffmat_t))
		sigmak = np.sum(sigmak, axis = 0)
		sigmak = sigmak / sum(rcut)
		sigma[i] = np.multiply(sigma[i], sigmak)

	return alpha, sigma, mu

def EM(k, x, epsilon):

	alpha, sigma, mu = EM_init(x,k)
	n, dim = np.shape(x)
	epoch = 0
	log_prev = 10000
	log_curr = 0
	while epoch < 1000 and abs(log_prev - log_curr) > epsilon:
		print('epoch: ' + str(epoch + 1))
		epoch = epoch + 1
		log_prev = log_curr
		alphaprev = alpha
		sigmaprev = sigma
		muprev = mu
		r = Estep(x, muprev, sigmaprev, alphaprev)
		alpha, sigma, mu = Mstep(x, r)
		# print(alpha)# print(sigma)# print(mu)

		alphamat = np.tile(alpha.transpose(), (n, 1))
		px = np.zeros([k, n])
		for j in range(0, k):
			px[j, :] = guassPDF(x, sigma[j], mu[j, :])
		px = px.transpose()

		p = px * alphamat
		p = np.sum(p, axis = 1)
		log_curr=np.sum(np.log(p))

		print('log likelihood: ')
		print('previous: ' + str(log_prev))
		print('now: ' + str(log_curr))
		print('')
	return alpha, sigma, mu, r

if __name__ == '__main__':
	data = np.load('./Train_Validation_set/rgbData.npy')
	data = data.transpose()    
	k = 3
	epsilon = 0.001
	alpha, sigma, mu, r = EM(k, data, epsilon)
	np.save('alpha_o.npy', alpha)
	np.save('sigma_o.npy', sigma)
	np.save('mu_o.npy', mu)