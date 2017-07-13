import numpy as np
import matplotlib.pyplot as plt
import os
from keras.preprocessing import image
from scipy.misc import imresize
from data_augmenter import ImageDataAugmenter

def mean_center_vgg(x):
    """
    Pre-processing provided by VGG researchers on VGG-Face
    """
    x[:, :, :, 0] -= 93.5940 #B
    x[:, :, :, 1] -= 104.7624 #G
    x[:, :, :, 2] -= 129.1863 #R

    return x


def preprocess_image(img, net='vgg'):
    """
    Convert image to proper format
    """
    if img.ndim < 4:
        img = np.expand_dims(img, axis=0)
    img = img[:, :, :, ::-1]
    if(net=='vgg'):
        img = mean_center_vgg(img)
    #if(net=='alexnet'):
    #    img = mean_center_alexnet(img)
    return img


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return [a[i] for i in p],[b[i] for i in p]


def data_generator(data, labels, batch_size, output_shape, n_output, augmenter=None, shuffle=True, net='vgg'):
    """
       Keras generator yielding batches of training/validation data.
       Applies data augmentation pipeline if `augment` is True.
       """
    while True:
        # Generate random batch of indices
        if shuffle:
            idxs = np.random.permutation(len(data))
        else:
            idxs = list(range(len(data)))

        id_data=0
        for batch in range(0, len(idxs), batch_size):
            batch_idxs = idxs[batch:(batch + batch_size)]
            # Output arrays
            batch_data = np.zeros((len(batch_idxs), output_shape[0], output_shape[1], output_shape[2]),
                                  dtype=np.float32)
            batch_labels = np.zeros((len(batch_idxs), n_output), dtype=np.float32)

            # Read in and preprocess a batch of images
            for id, i in enumerate(batch_idxs):
                # Read image
                # img = preprocess_image(plt.imread(data[i][..., :3])

                #new_img = data[i].thumbnail(output_shape, Image.ANTIALIAS)
                #print(new_img.shape)
                new_img = imresize(data[i],output_shape,'bilinear')
                #print(new_img.shape)
                x = image.img_to_array(new_img).astype(np.float32, copy=False)


                #x = image.img_to_array(image.load_img(data[i], target_size=output_shape)).astype(np.float32, copy=False)
                y = np.array(labels[i], copy=True)

                plt.imshow(x)#,plt.title(y+"-"+data[i])
                #print(y,data[i])
                plt.show()
                #plt.figure()
                #plt.imshow(x)
                #plt.show()
                #plt.subplot(121), plt.imshow(x), plt.title('Original')

                # Augment data
                if augmenter is not None:
                    assert (type(augmenter) is ImageDataAugmenter)
                    (x, var_que_no_se_usa_DANGER) = augmenter.augment(x, None)

                #plt.figure()
                #plt.imshow(x)
                #plt.show()
                #os.system("pause")

                img = preprocess_image(x, net)
                batch_data[id, :, :, :] = img
                batch_labels[id, :] = y
                id_data = id_data + 1

            yield (batch_data, batch_labels)


