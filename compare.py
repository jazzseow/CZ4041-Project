import numpy as np
import cv2
import glob
import time
import shutil
import os
import matplotlib.pyplot as plt
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


def find_pairs(cimg, cmask, cimg_case, imgs, masks, img_cases, cindex, matches):
	for i, (img, mask, img_case) in enumerate(zip(imgs, masks, img_cases)):
		if i != cindex and np.abs(cimg - img).sum() < 23000000 and (cmask.sum() != 0) == (mask.sum() == 0):
			matches.append([cimg_case, img_case])
	return matches

imgs, masks, img_cases = load_cv2_images()
matches = []
for j in range(47):
	for i, (img, mask, img_case) in enumerate(zip(imgs[j+1], masks[j+1], img_cases[j+1])):
		a= time.tiime()
		matches = find_pairs(img, mask, img_case, imgs[j+1], masks[j+1], img_cases[j+1], i, matches)
		print(time.time() - a)

print(len(matches))
'''
for pair in matches:
	os.remove(pair[1])
	shutil.copy2(pair[0], pair[1])
'''

