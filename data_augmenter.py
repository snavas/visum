import numpy as np
import scipy.ndimage
import PIL.Image
import cv2 as cv
#from keras.preprocessing.image import flip_axis

class ImageDataAugmenter(object):

    def __init__(self,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 rotation_prob = 0.75,
                 shift_prob = 0.75,
                 gaussian = None,
                 illumination = None,
                 zoom = None,
                 flip = None,
                 gamma = None,
                 contrast = None):
        self.dict = {'rotation_range': rotation_range,
                     'width_shift_range': width_shift_range,
                     'height_shift_range': height_shift_range}
        self.rotation_prob = rotation_prob
        self.shift_prob = shift_prob
        self.gaussian = None
        self.illumination = illumination
        self.zoom = zoom
        self.flip = flip
        self.gamma = gamma
        self.contrast = contrast

    def augment(self, x, y):
        #Flip
        #if self.flip is not None:
            #size_y = x.shape[1]
            #size_x = x.shape[0]
            # Vertical
            #if np.random.random() < self.flip:
                #x = flip_axis(x, 1)
                #y[0] = -y[0]
            # Horizontal
            #if np.random.random() < self.flip:
                #x = flip_axis(x, 0)
                #y[1] = -y[1]

        # Gamma
        if self.gamma is not None:
            random_gamma = np.random.uniform(self.gamma[0], self.gamma[1])
            image = x/255.0
            image = cv.pow(image, random_gamma)
            x = image*255

        # Illumination
        if self.illumination is not None:
            #image = cv.cvtColor(x,cv.COLOR_RGB2HSV)
            #random_bright = np.random.uniform(self.illumination[0], self.illumination[1])
            #image[:,:,2] = image[:,:,2]+random_bright*10
            #image[:,:,2] = np.clip(image[:,:,2],0.,255.)
            #image = cv.cvtColor(image,cv.COLOR_HSV2RGB)
            #x = image
            random_bright = np.random.uniform(self.illumination[0], self.illumination[1])
            #image = x/255.0
            image = cv.cvtColor(x,cv.COLOR_RGB2HSV)
            image[:,:,2] = cv.add(image[:,:,2], random_bright)
            image[:,:,2] = np.clip(image[:,:,2],0.,255.)
            #image = image*255
            x = cv.cvtColor(image,cv.COLOR_HSV2RGB)

        # Contrast
        if self.contrast is not None:
            #image = cv.cvtColor(x,cv.COLOR_RGB2HSV)
            #random_bright = np.random.uniform(self.illumination[0], self.illumination[1])
            #image[:,:,2] = image[:,:,2]+random_bright*10
            #image[:,:,2] = np.clip(image[:,:,2],0.,255.)
            #image = cv.cvtColor(image,cv.COLOR_HSV2RGB)
            image = x.copy()
            random_contrast = np.random.uniform(self.contrast[0], self.contrast[1])
            #image = x/255.0
            image[:,:,2] = image[:,:,2]*random_contrast
            image[:,:,2] = np.clip(image[:,:,2],0.,255.)
            #image = image*255
            x = image

        # Gaussian_noise
        if self.gaussian is not None:
            random_var = np.random.uniform(self.gaussian[0], self.gaussian[1])
            random_noise = np.random.randn(x.shape[0], x.shape[1])
            image = x.copy()
            for c in range(x.shape[2]):
                image[:,:,c] = np.clip(image[:,:,c] + random_var*127.5*random_noise, 0., 255.)
            x = image

        # Rotation
        if np.random.random() < self.rotation_prob and self.dict['rotation_range'] > 0.:
            theta = np.pi / 180 * np.random.uniform(-self.dict['rotation_range'], self.dict['rotation_range'])
        else:
            theta = 0

        # Translation - horizontal
        if np.random.random() < self.shift_prob and self.dict['width_shift_range'] > 0.:
            tx = np.random.uniform(-self.dict['width_shift_range'], self.dict['width_shift_range']) * x.shape[2]
        else:
            tx = 0

        # Translation - vertical
        if np.random.random() < self.shift_prob and self.dict['height_shift_range'] > 0.:
            ty = np.random.uniform(-self.dict['height_shift_range'], self.dict['height_shift_range']) * x.shape[1]
        else:
            ty = 0


        # Apply composition of transformations
        transf_mat = None

        # Zoom
        if self.zoom is not None:
            z = np.random.uniform(self.zoom[0], self.zoom[1])
            zoom_mat = np.array([[z, 0, 0],
                                 [0, z, 0],
                                 [0, 0, 1]])
            transf_mat = zoom_mat

        # Rotation
        if theta!= 0:
            rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
            transf_mat = rot_mat if transf_mat is None else np.dot(transf_mat, rot_mat)

        # Translation
        if tx != 0 or ty != 0:
            shift_mat = np.array([[1, 0, tx],
                                  [0, 1, ty],
                                  [0, 0, 1]])
            transf_mat = shift_mat if transf_mat is None else np.dot(transf_mat, shift_mat)

        if transf_mat is not None:
            # x = PIL.Image.fromarray(x)
            #size_y = x.shape[2]
            #size_x = x.shape[1]
            size_y = x.shape[1]
            size_x = x.shape[0]
            x = scipy.misc.toimage(x, cmin=0.0, cmax=255.0)
            x = x.transform((size_y, size_x), PIL.Image.AFFINE, np.float32(np.linalg.inv(transf_mat).flatten()), PIL.Image.BILINEAR)

            if y is not None:
                y = np.hstack((y, np.ones((5,1))))
                y = np.dot(transf_mat, np.transpose(y))
                y = np.transpose(np.delete(y, 2, 0))

        if y is not None:
            y = y.flatten()
            #for i in range(len(y)):

            #    y1 = np.dot(transf_mat, y[i])
             #   y = np.dot(transf_mat,np.expand_dims(y, axis = 2))
        return (x,y)
