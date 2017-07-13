import numpy as np
import os
from keras.preprocessing import image

def data_generator(data, labels, batch_size, output_shape, n_output, augmenter=None, shuffle=True):
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
                                  dtype=np.float)
            batch_labels = np.zeros((len(batch_idxs), n_output), dtype=np.float32)

            # Read in and preprocess a batch of images
            for id, i in enumerate(batch_idxs):

                # Determine num of images in folder
                images = [each for each in os.listdir(data[i]) if each.endswith('.png')]
                # Expand dims (dim [1]) for n_frames
                batch_data = np.expand_dims(batch_data, axis=1)
                #batch_data[0,:,:,:] = \
                loaded_images = [image.img_to_array(image.load_img(os.path.join(data[i], f),
                                       target_size=output_shape)).astype(np.float, copy=False)
                                       for f in images]
                seq_images = np.zeros((len(batch_idxs), len(images), output_shape[0], output_shape[1], output_shape[2]),
                                  dtype=np.float)
                seq_images[id] = loaded_images
                batch_data = seq_images
                y = np.array(labels[i], copy=True)

                # Augment data
                #if augmenter is not None:
                #    assert (type(augmenter) is ImageDataAugmenter)
                #    (x, _) = augmenter.augment(x, None)

                #img = preprocess_image(x, net)
                #batch_data[id, :, :, :] = img
                batch_labels[id, :] = y
                id_data = id_data + 1

            yield (batch_data, batch_labels)