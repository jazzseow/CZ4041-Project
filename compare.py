import numpy as np
import cv2
import glob
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

def find_pairs(cimg, cmask, cid, imgs, masks, img_cases, cindex, matches):
	for i, (img, mask, img_case) in enumerate(zip(imgs, masks, img_cases)):
		if np.abs(cimg - img).sum() < 23000000 and i != cindex and (cmask.sum() == 0) != (mask.sum() == 0):
			matches.append((cimg, cmask, cid, img, mask, img_case))
	return matches

imgs, masks, img_cases = load_cv2_images()
matches = []
for j in range(47):
	for i, (img, mask, img_case) in enumerate(zip(imgs[j+1], masks[j+1], img_cases[j+1])):
		matches = find_pairs(img, mask, img_case, imgs[j+1], masks[j+1], img_cases[j+1], i, matches)

repeats, unique = [], []
for i, m in enumerate(matches):

    # Using pixel sums as an ID for the picture
    if m[0].sum() not in repeats\
    or m[3].sum() not in repeats:

        unique.append(m[0].sum())
        fig, ax = plt.subplots(2, 2)
        if m[1].sum() == 0:
            i1, i2 = 1, 0
        else:
            i1, i2 = 0, 1
		print m[2]
        ax[i1][0].imshow(m[0], cmap='hot')
        ax[i1][0].set_title(m[2])
        ax[i1][1].imshow(m[1], cmap='hot')
        ax[i1][1].set_title(m[2][:-4]+'_mask.tif')

        ax[i2][0].imshow(m[3], cmap='hot')
        ax[i2][0].set_title(m[5])
        ax[i2][1].imshow(m[4], cmap='hot')
        ax[i2][1].set_title(m[5][:-4]+'_mask.tif')

        fig.subplots_adjust(hspace=0.4)
        plt.show()

    repeats.append(m[0].sum())
    repeats.append(m[3].sum())
    if i == 98:
        break
