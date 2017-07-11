# Author: Cristina Palmero


from keras.preprocessing.image import random_rotation, random_shift, flip_axis, apply_transform, transform_matrix_offset_center

import numpy as np
import cv2 as cv

#def flip_axis(x, axis):
#    x = np.asarray(x).swapaxes(axis, 0)
#    x = x[::-1, ...]
#    x = x.swapaxes(0, axis)
#    return x


'''
Random shadow and brightness augmentation from
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
'''

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv.cvtColor(image,cv.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2)==1:
        random_bright = .4+.7*np.random.uniform()
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
        image_hls[:,:,1] = np.clip(image_hls[:,:,1],0.,255.)
    image = cv.cvtColor(image_hls,cv.COLOR_HLS2RGB)
    image = np.clip(image,0.,255.)

    return image

def modify_illumination(image):
    image1 = cv.cvtColor(image,cv.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2] = np.clip(image1[:,:,2],0.,255.)
    image1 = cv.cvtColor(image1,cv.COLOR_HSV2RGB)
    return image1

def add_gaussian_noise(image, var):
     random_var = np.random.uniform(var[0], var[1])
     shape = list(image.shape)
     random_noise = np.random.randn(shape[0], shape[1])
     image1 = image
     for c in range(shape[2]):
         image1[:,:,c] = np.clip(image1[:,:,c] + random_var*127.5*random_noise, 0., 255.)

     return image1


def add_occlusions(image, prob_range, rect_size_range, squared):
    p = np.random.uniform(prob_range[0], prob_range[1])
     
    image_shape = list(image.shape)
    image_area = image_shape[0] * image_shape[1]
    occluded_image_area = np.int(p * image_area)

    if squared:
        rect_size_w = rect_size_h = np.random.randint(rect_size_range[0]*image_shape[0], rect_size_range[1]*image_shape[1])
    else:
        rect_size_w = np.random.randint(rect_size_range[0]*image_shape[0], rect_size_range[1]*image_shape[1])
        rect_size_h = np.random.randint(rect_size_range[0]*image_shape[0], rect_size_range[1]*image_shape[1])
     
    rect_area = rect_size_h * rect_size_w

    if rect_area != 0:
        n_patches = occluded_image_area // rect_area

        for patch in range(n_patches):
            pt1 = (np.random.randint(0, image_shape[1]), np.random.randint(0, image_shape[0]))
            pt2 = (np.clip(pt1[0] + rect_size_w, 0, image_shape[1]), np.clip(pt1[1] + rect_size_h, 0, image_shape[0]))
            
            for c in range(image_shape[2]):
                 image[pt1[1]:pt2[1],pt1[0]:pt2[0],c] = 0.

    return image


def random_zoom(image, zoom_range, row_axis=1, col_axis=2, channel_axis=0,
                fill_mode='nearest', cval=0.):
    '''
    Adapted from keras.preprocessing.image to force transformation to apply same zoom to both axes
    ''' 
    if zoom_range[0] == 1 and zoom_range[1] == 1:
        z = 1
    else:
        z = np.random.uniform(zoom_range[0], zoom_range[1])

    zoom_matrix = np.array([[z, 0, 0],
                            [0, z, 0],
                            [0, 0, 1]])

    h, w = image.shape[row_axis], image.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    image = apply_transform(image, transform_matrix, channel_axis, fill_mode, cval)
    return image


class ImageDataAugmenter(object):

    def __init__(self,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 zoom_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 occlusion_area_range=0.,
                 occlusion_size_range=0.,
                 squared_occlusion=False,
                 gaussian_noise_range=0.,
                 saltnpepper_noise_range=0.,
                 illumination = False,
                 shadow = False,
                 rotation_prob = 0.75,
                 shift_prob = 0.75,
                 zoom_prob = 0.75,
                 flip_prob = 0.5,
                 illumination_prob = 0.75,
                 shadow_prob = 0.75,
                 gaussian_noise_prob = 0.75,
                 occlusion_prob = 0.75):

        self.fill_mode = fill_mode
        self.cval = cval
        self.rotation_prob = rotation_prob
        self.shift_prob = shift_prob
        self.zoom_prob = zoom_prob
        self. flip_prob = flip_prob
        self.illumination_prob = illumination_prob
        self.shadow_prob = shadow_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.occlusion_prob = occlusion_prob

        self.dict = {'rotation_range': rotation_range, 
                     'width_shift_range': width_shift_range, 'height_shift_range': height_shift_range,
                     'zoom_range': zoom_range, 
                     'horizontal_flip': horizontal_flip, 'vertical_flip':vertical_flip,
                     'gaussian_noise_range': gaussian_noise_range, 'saltnpepper_noise_range': saltnpepper_noise_range, 
                     'illumination': illumination, 
                     'occlusion_area_range': occlusion_area_range, 'occlusion_size_range': occlusion_size_range,
                     'squared_occlusion': squared_occlusion, 'shadow': shadow}

        self.check_ranges('zoom_range', zoom_range)
        self.check_ranges('gaussian_noise_range', gaussian_noise_range)
        self.check_ranges('occlusion_area_range', occlusion_area_range)
        self.check_ranges('occlusion_size_range', occlusion_size_range)



    def augment(self, x, y):

        #Rotation
        if np.random.random() < self.rotation_prob and self.dict['rotation_range'] > 0.:
            theta = np.pi / 180 * np.random.uniform(-self.dict['rotation_range'], self.dict['rotation_range'])
            #x = random_rotation(x, self.dict['rotation_range'], row_axis = 0, col_axis = 1, channel_axis = 2,
            #            fill_mode = self.fill_mode, cval = self.cval)
        else:
            theta = 0

        #Translation in y
        if np.random.random() < self.shift_prob and self.dict['height_shift_range'] > 0.:
            #x = random_shift(x, 0., self.dict['height_shift_range'], row_axis = 0, col_axis = 1,
            #                channel_axis = 2, fill_mode = self.fill_mode, cval = self.cval)
            tx = np.random.uniform(-self.dict['height_shift_range'], self.dict['height_shift_range']) * x.shape[0]
        else:
            tx = 0

        #Translation in x
        if np.random.random() < self.shift_prob and self.dict['width_shift_range'] > 0.:
            #x = random_shift(x, self.dict['width_shift_range'], 0., row_axis = 0, col_axis = 1,
            #                channel_axis = 2, fill_mode = self.fill_mode, cval = self.cval)
            ty = np.random.uniform(-self.dict['width_shift_range'], self.dict['width_shift_range']) * x.shape[1]
        else:
            ty = 0
        
        #Zoom
        if np.random.random() < self.zoom_prob and self.dict['zoom_range'][1] > 0.:
            #x = random_zoom(x, self.dict['zoom_range'], row_axis = 0, col_axis = 1, channel_axis = 2, 
            #            fill_mode = self.fill_mode, cval = self.cval)
            if self.dict['zoom_range'][0] == 1 and self.dict['zoom_range'][1] == 1:
                z = 1
            else:
                z = np.random.uniform(self.dict['zoom_range'][0], self.dict['zoom_range'][1])
        else:
            z = 1

        #Apply composition of transformations
        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if z != 1:
            zoom_matrix = np.array([[z, 0, 0],
                                    [0, z, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[0], x.shape[1]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, channel_axis = 2,
                                fill_mode=self.fill_mode, cval=self.cval)

        #Vertical flip
        if self.dict['vertical_flip'] == True:
            if np.random.random() < self.flip_prob:
                x = flip_axis(x, 0)
                y[1] = -y[1]

        #Horizontal flip
        if self.dict['horizontal_flip'] == True:
            if np.random.random() < self.flip_prob:
                x = flip_axis(x, 1)
                y[0] = -y[0]

        #Illumination
        if self.dict['illumination']:
            if np.random.random() < self.illumination_prob:
                x = modify_illumination(x)
        
        #Shadow
        if self.dict['shadow']:
            if np.random.random() < self.shadow_prob:
                x =  add_random_shadow(x)
         
        #Additive gaussian noise           
        if self.dict['gaussian_noise_range'][1] > 0.:
            if np.random.random() < self.gaussian_noise_prob:
                x = add_gaussian_noise(x, self.dict['gaussian_noise_range'])

        #Salt and pepper noise
        if self.dict['saltnpepper_noise_range'] > 0.:
            ...

        #Occlusion
        if self.dict['occlusion_area_range'][1] > 0.:
            if np.random.random() < self.occlusion_prob:
                add_occlusions(x, self.dict['occlusion_area_range'], self.dict['occlusion_size_range'], self.dict['squared_occlusion'])

        return (x, y)


    def check_ranges(self, param_name, param_value):

        if np.isscalar(param_value):
            if param_value < 0. or param_value > 1.:
                    raise ValueError((param_name, ' should be within range [0,1]'))
            else:
                if param_name == 'zoom_range':
                    self.dict[param_name] = [1 - param_value, 1 + param_value]
                else:
                    self.dict[param_name] = [0., param_value]

        elif len(param_value) == 2:

            if  param_name != 'zoom_range' and (param_value[0] < 0. or param_value[0] > 1. or param_value[1] < 0. or param_value[1] > 1.):
                    raise ValueError((param_name, ' should be within range [0,1]'))
            else:
                self.dict[param_name] = [param_value[0], param_value[1]]
        else:
            raise ValueError((param_name, ' should be a float or '
                                'a tuple or list of two floats. '
                                'Received arg: '), param_value)

