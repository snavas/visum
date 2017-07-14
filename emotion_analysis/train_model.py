#!/usr/bin/env python

# ==============================================================================
# Imports
# ==============================================================================
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.engine import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import pickle
from vggface import define_model
from keras.preprocessing import image
from scipy.misc import imresize
from sklearn.model_selection import KFold

# TODO change folder structure so that common files are in root/
from data_augmenter import ImageDataAugmenter
from utils import *

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import sklearn

## Settings from config file
sys.path.insert(0, 'src/')
import config

config.init()

## emotions code
__emotions__ = config.__emotions__

## Import "Feature extraction" functions
#sys.path.insert(0, 'src/feat_extraction/geometry/')
#from feat_kpts_geometry import *

#sys.path.insert(0, 'src/feat_extraction/appearance/lbp/')

## Import load_data functions from load_data.py
from load_data import load_csv, load_test_labels

if __name__ == "__main__":
    # ==============================================================================
    # Load Dataset
    # ==============================================================================
    train_fn = 'data/training.csv'
    test_fn = 'data/test.csv'

    ## Training data
    train_X, train_kpts, train_Y = load_csv(train_fn, istrain=True)
    print(train_X.shape, train_kpts.shape, train_Y.shape)

    new_train_X = np.empty((train_X.shape[0],train_X.shape[1],train_X.shape[2],3))
    for i, pic in enumerate(train_X):
        new_pic = np.stack((pic,)*3)
        new_pic = new_pic.transpose(1,2,0)
        new_train_X[i] = new_pic

    print(new_train_X.shape)

    ## Test data
    X_test, kpts_test = load_csv(test_fn, istrain=False)
    print(X_test.shape, kpts_test.shape)
    new_X_test = np.empty((X_test.shape[0],X_test.shape[1],X_test.shape[2],3))
    for i, pic in enumerate(X_test):
        new_pic = np.stack((pic,)*3)
        new_pic = new_pic.transpose(1,2,0)
        new_X_test[i] = new_pic

    # VGG parameters
    batch_size = 32
    output_shape = (224, 224, 3)
    n_output = 8
    epochs = 30
    lr = 0.0001
    dropout = 0.3

    # Define model
    model = define_model(n_output, lr, dropout)

    # TODO add data augmenter here

    # TODO create new folder for checkpoints
    # Checkpoint model
    filepath = "weights-improvement-{epoch:03d}-{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

    # Early stopping
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=1, mode='min')

    # List of callbacks to add to fit function
    callbacks_list = [checkpoint, earlystopping]

    augmenter = ImageDataAugmenter(rotation_range=0,
                                   rotation_prob=0,
                                   height_shift_range=0,
                                   width_shift_range=0,
                                   gaussian=[0,0.05],
                                   illumination=[-25, 25],
                                   zoom=[1, 1.01],
                                   flip=0.5,
                                   #gamma=[0, 0.01],
                                   gamma=None,
                                   blur=0.2,
                                   contrast=[-15, 15])

    kf = KFold(n_splits=5)
    kf.get_n_splits()
    for train_i, validation_i in kf.split(new_train_X):
        x_train, x_validation = new_train_X[train_i], new_train_X[validation_i]
        y_train, y_validation = train_Y[train_i], train_Y[validation_i]

    # Fit model and save statistics in hist
    hist = model.fit_generator(
        data_generator(x_train, y_train, batch_size, output_shape, n_output, augmenter=augmenter, net='vgg'),
        steps_per_epoch=np.ceil(len(x_train) / batch_size), epochs=epochs,
        ####### NEW
        validation_data = data_generator(x_validation, y_validation, batch_size, output_shape, n_output, augmenter=None, net='vgg'),
        validation_steps = np.ceil(len(x_validation) / batch_size), callbacks = callbacks_list
    )

    #hist = model.fit_generator(
    #    data_generator(new_train_X, train_Y, batch_size, output_shape, n_output, augmenter=augmenter, net='vgg'),
    #    steps_per_epoch=np.ceil(len(train_X) / batch_size), epochs=epochs)


    model.save('algo.hd5')

    # with open('hist_algo.pickle', 'wb') as f:
    #    pickle.dump(hist.history, f)

    # Test
    val_x = [image.img_to_array(imresize(fname, output_shape, 'bilinear')) for fname in new_X_test]
    # val_x = [plt.imread(fname)[..., :3] for fname in test_images]
    val_x = np.array(val_x)
    # val_x = np.expand_dims(val_x, axis=0)
    val_x = val_x.astype(np.float32, copy=False)
    val_x = val_x[:, :, :, ::-1]
    val_x[:, :, :, 0] -= 93.5940
    val_x[:, :, :, 1] -= 104.7624
    val_x[:, :, :, 2] -= 129.1863
    # val_x = preprocess_image(val_x)
    print(val_x.shape)

    preds = model.predict(val_x)

    # ==============================================================================
    # MAKE A SUBMISSION
    # ==============================================================================
    header = [str(idx) + ": " + emo for idx, emo in enumerate(__emotions__)]
    header.insert(0, "image_id")
    output = [','.join(header)]
    for i in range(0, len(val_x)):
        output.append(str(i) + ',' + ','.join(map(str, preds[i])))

    if len(os.sys.argv) == 1:
        outf = "my_predictions_vgg.csv"
    else:
        outf = os.sys.argv[1]

    f = open(outf, 'w')
    f.write('\n'.join(output))
    f.close()

    print("Predictions written to '{}'".format(outf))


    #if len(os.sys.argv) == 1:
    #    outf = "my_predictions_vgg.csv"
    #else:
    #    outf = os.sys.argv[1]

    fids = np.arange(1, len(preds) + 1).reshape(len(preds), 1)
    #preds = np.hstack([fids, preds])

    # write submission file
    #fmt = '%.1f,' + ','.join(['%.6f'] * 8)
    #np.savetxt(outf, preds, fmt=fmt, header="image_id,0: neutral,1: anger,2: contempt,3: disgust,4: fear,5: happy,6: sadness,7: surprise", delimiter=",")

    #print("\nPredictions written to '{}'".format(outf))