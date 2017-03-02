import numpy as np
import cv2
import glob
import time
import shutil
import os
from natsort import natsorted

def load_cv2_images():

	imgs, masks, img_cases = {}, {}, {}

	for i in range(47):
		imgs[i+1] = []
		masks[i+1] = []
		img_cases[i+1] = []

	paths = glob.glob('*.tif')

	paths = [p for p in paths if 'mask' not in p]
	paths = natsorted(paths, key=lambda y: y.lower())

	for p in paths:
		patient_id = int(p.split('_')[0])

		imgs[patient_id].append(cv2.imread(p, 0))
		masks[patient_id].append(cv2.imread(p[:-4]+'_mask.tif', 0))
		img_cases[patient_id].append(p)

	for i in range(47):
		imgs[i+1] = np.array(imgs[i+1])
		masks[i+1] = np.array(masks[i+1])

	return imgs, masks, img_cases


def add_white_find_pairs(imgs, masks, img_cases):
	matches = []
	for j in range(47):
		for cindex, (cimg, cmask, cimg_case) in enumerate(zip(imgs[j+1], masks[j+1], img_cases[j+1])):
			for i, (img, mask, img_case) in enumerate(zip(imgs[j+1], masks[j+1], img_cases[j+1])):
				if np.abs(cimg - img).sum() < 23000000 and i != cindex and \
				cmask.sum() != 0 and mask.sum() == 0 and \
				([img_case, cimg_case] not in matches):
					matches.append([cimg_case, img_case])
	
	# get only unique path destinations
	unique_matches = []
	flag = True
	for pair in matches:
		for unique_pair in unique_matches:
			if pair[1] == unique_pair[1]:
				flag = False
				break
		if flag == True:
			unique_matches.append(pair)
		else:
			flag = True
			
	for pair in unique_matches:
		os.remove(pair[1])
		shutil.copy2(pair[0], pair[1])

def add_black_find_pairs(imgs, masks, img_cases):
	matches = []
	for j in range(47):
		for cindex, (cimg, cmask, cimg_case) in enumerate(zip(imgs[j+1], masks[j+1], img_cases[j+1])):
			for i, (img, mask, img_case) in enumerate(zip(imgs[j+1], masks[j+1], img_cases[j+1])):
				if np.abs(cimg - img).sum() < 23000000 and i != cindex and \
				cmask.sum() == 0 and mask.sum() != 0 and \
				([img_case, cimg_case] not in matches):
					matches.append([cimg_case, img_case])
	
	# get only unique path destinations
	unique_matches = []
	flag = True
	for pair in matches:
		for unique_pair in unique_matches:
			if pair[1] == unique_pair[1]:
				flag = False
				break
		if flag == True:
			unique_matches.append(pair)
		else:
			flag = True
	
	for pair in unique_matches:
		os.remove(pair[1])
		shutil.copy2(pair[0], pair[1])
		
def destroy_find_pairs(imgs, masks, img_cases):
	matches = []
	for j in range(47):
		for cindex, (cimg, cmask, cimg_case) in enumerate(zip(imgs[j+1], masks[j+1], img_cases[j+1])):
			for i, (img, mask, img_case) in enumerate(zip(imgs[j+1], masks[j+1], img_cases[j+1])):
				if np.abs(cimg - img).sum() < 23000000 and i != cindex and \
				cmask.sum() == 0 and mask.sum() != 0:
					matches.append([cimg_case, img_case])
	
	# flatten list of list
	flat = [element for pair in matches for element in pair]
	unique_matches = []
	for element in flat:
		if element not in unique_matches:
			unique_matches.append(element)

	for pair in unique_matches:
		os.remove(pair)
	

imgs, masks, img_cases = load_cv2_images()
destroy_find_pairs(imgs, masks, img_cases)

