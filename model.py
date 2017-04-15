import os
import numpy as np
from cv2 import Canny
from scipy.ndimage import gaussian_filter as sharp
from skimage.transform import resize
from skimage.io import imread
from skimage.exposure import equalize_hist as contrast
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras import backend as K
from itertools import chain
from matplotlib import pyplot as plt

K.set_image_data_format('channels_last')

# hyperparameters
input_dim = (96, 96, 1)
conv_depth = [1, 32, 64, 128, 256, 512] # first index reserved for final output
filter_dim = ['nil', (1, 1), (2, 2), (3, 3), (4, 4)] # index refers directly to filter size
#pool_dim = (4, 4)
learning_rate = 1e-5
batch = 32
epochs = 3
activation = ['sigmoid', 'relu'] # first index reserved for final output
zero_pad = 'same'

# fix random seed for reproducibility and initialise dataset
np.random.seed(1)

def preprocess():
	image_names_train = os.listdir('ultrasound/train/')
	image_names_test = os.listdir('ultrasound/test/')

	img_dim_train = (len(image_names_train) / 2, 420, 580)
	img_dim_test = (len(image_names_test), 420, 580)

	imgs = np.ndarray(img_dim_train, dtype=np.uint8)
	imgs_mask = np.ndarray(img_dim_train, dtype=np.uint8)
	imgs_test = np.ndarray(img_dim_test, dtype=np.uint8)
	imgs_id = np.ndarray((len(image_names_test), ), dtype=np.int32)

	i = 0
	for image_name in image_names_train:
		if 'mask' in image_name: continue
		image_mask_name = image_name.split('.')[0] + '_mask.tif'
		imgs[i] = contrast(np.array([imread(os.path.join('ultrasound/train/', image_name), as_grey=True)]))
		imgs_mask[i] = np.array([imread(os.path.join('ultrasound/train/', image_mask_name), as_grey=True)])
		i += 1

	i = 0
	for image_test_name in image_names_test:
		imgs_id[i] = int(image_test_name.split('.')[0])
		imgs_test[i] = contrast(np.array([imread(os.path.join('ultrasound/test/', image_test_name), as_grey=True)]))
		i += 1

	X_train = np.ndarray((imgs.shape[0], input_dim[0], input_dim[1]), dtype=np.uint8)
	y_train = np.ndarray((imgs_mask.shape[0], input_dim[0], input_dim[1]), dtype=np.uint8)
	X_test = np.ndarray((imgs_test.shape[0], input_dim[0], input_dim[1]), dtype=np.uint8)
	for i in range(imgs.shape[0]):
		X_train[i] = resize(imgs[i], input_dim[0:2], preserve_range=True)
		y_train[i] = resize(imgs_mask[i], input_dim[0:2], preserve_range=True)
	for i in range(imgs_test.shape[0]):
		X_test[i] = resize(imgs_test[i], input_dim[0:2], preserve_range=True)

	X_train = X_train[..., np.newaxis]
	y_train = y_train[..., np.newaxis]
	X_test = X_test[..., np.newaxis]

	X_train = X_train.astype('float32')
	y_train = y_train.astype('float32')
	X_test = X_test.astype('float32')

	mean = np.mean(X_train)
	std = np.std(X_train)

	X_train = (X_train - mean) / std
	y_train /= 255.
	X_test = (X_test - mean) / std

	return X_train, y_train, X_test, imgs_id

# the dice coefficient calculation
def dice_loss(ground_truth, prediction):
    ground_truth = K.flatten(ground_truth)
    prediction = K.flatten(prediction)
    return -(2. * K.sum(ground_truth * prediction) + 1) / (K.sum(ground_truth) + K.sum(prediction) + 1)

def model(X_train, y_train, X_test):
	inputs = Input(input_dim)
	conv1 = Conv2D(conv_depth[1], filter_dim[3], activation=activation[1], padding=zero_pad)(inputs)
	conv1 = Conv2D(conv_depth[1], filter_dim[3], activation=activation[1], padding=zero_pad)(conv1)
	pool1 = MaxPooling2D(pool_size=filter_dim[2])(conv1)
	conv2 = Conv2D(conv_depth[2], filter_dim[3], activation=activation[1], padding=zero_pad)(pool1)
	conv2 = Conv2D(conv_depth[2], filter_dim[3], activation=activation[1], padding=zero_pad)(conv2)
	'''
	pool2 = MaxPooling2D(pool_size=filter_dim[2])(conv2)
	conv3 = Conv2D(conv_depth[3], filter_dim[3], activation=activation[1], padding=zero_pad)(pool2)
	conv3 = Conv2D(conv_depth[3], filter_dim[3], activation=activation[1], padding=zero_pad)(conv3)
	pool3 = MaxPooling2D(pool_size=filter_dim[2])(conv3)
	conv4 = Conv2D(conv_depth[4], filter_dim[3], activation=activation[1], padding=zero_pad)(pool3)
	conv4 = Conv2D(conv_depth[4], filter_dim[3], activation=activation[1], padding=zero_pad)(conv4)
	pool4 = MaxPooling2D(pool_size=filter_dim[2])(conv4)
	conv5 = Conv2D(conv_depth[5], filter_dim[3], activation=activation[1], padding=zero_pad)(pool4)
	conv5 = Conv2D(conv_depth[5], filter_dim[3], activation=activation[1], padding=zero_pad)(conv5)
	unpool4 = concatenate([UpSampling2D(size=filter_dim[2])(conv5), conv4], axis=3)
	deconv5 = Conv2D(conv_depth[4], filter_dim[3], activation=activation[1], padding=zero_pad)(unpool4)
	deconv5 = Conv2D(conv_depth[4], filter_dim[3], activation=activation[1], padding=zero_pad)(deconv5)
	unpool3 = concatenate([UpSampling2D(size=filter_dim[2])(deconv5), conv3], axis=3)
	deconv4 = Conv2D(conv_depth[3], filter_dim[3], activation=activation[1], padding=zero_pad)(unpool3)
	deconv4 = Conv2D(conv_depth[3], filter_dim[3], activation=activation[1], padding=zero_pad)(deconv4)
	unpool2 = concatenate([UpSampling2D(size=filter_dim[2])(deconv4), conv2], axis=3)
	deconv3 = Conv2D(conv_depth[2], filter_dim[3], activation=activation[1], padding=zero_pad)(unpool2)
	deconv3 = Conv2D(conv_depth[2], filter_dim[3], activation=activation[1], padding=zero_pad)(deconv3)
	'''
	unpool1 = concatenate([UpSampling2D(size=filter_dim[2])(conv2), conv1], axis=3)
	deconv2 = Conv2D(conv_depth[1], filter_dim[3], activation=activation[1], padding=zero_pad)(unpool1)
	deconv2 = Conv2D(conv_depth[1], filter_dim[3], activation=activation[1], padding=zero_pad)(deconv2)
	deconv1 = Conv2D(conv_depth[0], filter_dim[1], activation=activation[0])(deconv2)
	model = Model(inputs=[inputs], outputs=[deconv1])
	model.compile(optimizer=Adam(lr=learning_rate), loss=dice_loss, metrics=['accuracy'])

	# train model
	model.fit(X_train, y_train, batch_size=batch, nb_epoch=epochs, verbose=1, shuffle=True, validation_split=0.2)

	# output prediction with test data
	return model.predict(X_test, verbose=1)

def save(predict, ids):
	file = open('submission.csv', 'w+')
	file.write('img,pixels\n')

	argsort = np.argsort(ids)
	ids = ids[argsort]
	predict = predict[argsort]

	total = predict.shape[0]
	for i in range(total):
		img = predict[i, 0]
		img = img.astype('float32')
		img = (img > 0.5).astype(np.uint8)
		img = resize(img, (580, 420), preserve_range=True)

		x = img.transpose().flatten()
		y = np.where(x > 0)[0]
		if len(y) < 10:
			file.write(str(ids[i]) + ',\n')
			continue
		z = np.where(np.diff(y) > 1)[0]
		start = np.insert(y[z+1], 0, y[0])
		length = np.append(y[z], y[-1]) - start
		res = list(chain.from_iterable([[s+1, l+1] for s, l in zip(list(start), list(length))]))
		pixel = ' '.join([str(r) for r in res])
		file.write(str(ids[i]) + ',' + pixel + '\n')

X_train, y_train, X_test, id = preprocess() # load the preprocessed dataset
predict = model(X_train, y_train, X_test) # get prediction
save(predict, id) # output prediction in csv
