import numpy as np
import PIL
import scipy.misc
import os
import matplotlib.pyplot as plt

from data_augmenter import *

def load_image(filename):
	img = PIL.Image.open(filename)
	img = np.asarray(img, dtype='uint8')  # output is yxc
	#img = np.transpose(img, (2, 0, 1))  # yxc -> cyx?
	return img


def save_image(filename, img):
	#img = np.transpose(img, (1, 2, 0))
	scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(filename)


def visualize_image(img, ldmks):
	plt.figure()
	img_s = scipy.misc.toimage(img, cmin=0.0, cmax=255.0)
	plt.imshow(img_s)
	ldmks_c = np.asarray(ldmks, np.uint8)
	plt.scatter(x=[ldmks_c[0::2]],y=[ldmks_c[1::2]], c='r')
	plt.show()


if __name__ == "__main__":

	# Load ground truth
	gt_filename = 'list_landmarks_celeba_crop_10k.txt'
	data_folder = 'img_celeba_crop_10k'
	gt_file = open(gt_filename, 'rb')

	gt_imagename = np.genfromtxt(gt_filename, delimiter=' ', dtype='U12', usecols=0)
	gt_landmarks = np.loadtxt(gt_filename, delimiter=' ', usecols=range(1,11))

	assert(len(gt_imagename) == len(gt_landmarks))

	augmenter = ImageDataAugmenter(rotation_range = 45, rotation_prob=50,
								   height_shift_range= 0.2, width_shift_range=0.3, 
								   gaussian=[0,0.3], illumination=None, zoom=[0.85,1.15],
								    flip=0.25, gamma=None, contrast=[-50,50])

	# Load and visualize images
	for i in range(len(gt_imagename)):
		img = load_image(os.path.join(data_folder, gt_imagename[i]))
		#### START VISUALIZATION
		img_s = scipy.misc.toimage(img, cmin=0.0, cmax=255.0)
		plt.subplot(121),plt.imshow(img_s),plt.title('Input')
		ldmks_c = np.asarray(gt_landmarks[i], np.uint8)
		plt.scatter(x=[ldmks_c[0::2]],y=[ldmks_c[1::2]], c='r')
		#visualize_image(img, gt_landmarks[i])
		#### END VISUALIZATION
		#x = np.array(img, copy= True)
		y = np.array(gt_landmarks[i], copy = True)
		print("original gt: ", y)
		x_aug, y_aug = augmenter.augment(img, y.reshape((5, 2)))
		y_aug = np.round(y_aug) # negative values will be visible outside of the image
		print("augmented gt: ", y_aug)
		### START VISUALIZATION
		img_s2 = scipy.misc.toimage(x_aug, cmin=0.0, cmax=255.0)
		plt.subplot(122),plt.imshow(img_s2),plt.title('Input')
		ldmks_c2 = np.asarray(y_aug, np.uint8)
		plt.scatter(x=[ldmks_c2[0::2]],y=[ldmks_c2[1::2]], c='r')
		plt.show()
		### END VISUALIZATION
		#visualize_image(x_aug, y_aug)
		os.system("pause")

