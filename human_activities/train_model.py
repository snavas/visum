#!/usr/bin/env python
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import cv2
import os
import sys
import pickle

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing import image

from resnet import define_model
from utils import *

def read_video(path):
    frames = []
    for img_filename in sorted(list(os.walk(path))[0][2]):
        frames.append(cv2.imread(os.path.join(path, img_filename), 0))
    return np.asarray(frames)


def one_hot_encoder(ids, classes, rows, cols):
    class_dict = dict(enumerate(classes))
    classes_dict = {v: k for k, v in class_dict.items()}

    onehot_y = np.zeros((rows, cols))
    for i, item in enumerate(ids):
        onehot_y[i, classes_dict[np.str(item)]] = 1
    return onehot_y


if __name__ == '__main__':

    n_output = 6
    lr = 0.0001
    dropout = 0.3
    epochs = 30
    batch_size = 1
    output_shape = (200, 200, 3)

    # Load
    f = open('sets/training.csv')
    lines = np.asarray(list(csv.reader(f)))
    f.close()

    print('Training file loaded.')

    # lines = lines[: 100]

    videos = lines[:, 0]
    train_X = [os.path.join(os.getcwd(), video) for video in videos]
    train_X = list(map(lambda s: s.replace('/', '\\'), train_X))
    labels = lines[:, 1]
    classes = sorted(list(set(labels)))
    train_Y = one_hot_encoder(labels, classes, len(labels), len(classes))

    # Just pick 300 videos
    #train_X = train_X[:300]
    #train_Y = train_Y[:300]

    features = []

    f = open('sets/sample_submission.csv')
    lines = list(csv.reader(f))
    f.close()

    #test_X = lines[1:,0]
    #classes = lines[0][1:]
    #index_ = [list(model.classes_).index(c) for c in classes]

    # Define model
    model = define_model(n_output, lr, dropout)

    # TODO add data augmenter here

    # TODO create new folder for checkpoints
    # Checkpoint model
    filepath = "weights-improvement-{epoch:03d}-{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

    # Early stopping
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=40, verbose=1, mode='min')

    # List of callbacks to add to fit function
    callbacks_list = [checkpoint, earlystopping]

    #frame_sequences
    #for i, v in enumerate(videos):
    #    frames = read_video(v)

    # Fit model and save statistics in hist
    hist = model.fit_generator(
        data_generator(train_X, train_Y, batch_size, output_shape, n_output, augmenter=None),
        steps_per_epoch=len(train_X), epochs=epochs)

    #model.save('algo.hd5')

    with open('hist_algo.pickle', 'wb') as f:
        pickle.dump(hist.history, f)

    # Test
    test = lines[1:]
    #test = test[:100]


    output = [','.join(lines[0])]
    for i, seq in enumerate(test):
        seq_path = os.path.join(os.getcwd(), seq[0])
        #seq_path = np.str(map(lambda s: s.replace('/', '\\'), seq_path))
        seq_path = seq_path.replace('/', '\\')
        images = [each for each in os.listdir(seq_path) if each.endswith('.png')]

        loaded_images = [image.img_to_array(image.load_img(os.path.join(seq[0], f),
                                                       target_size=output_shape)).astype(np.float, copy=False)
                         for f in images]
        seq_images = np.zeros((1, len(loaded_images), output_shape[0], output_shape[1], output_shape[2]), dtype=np.float32)
        seq_images[0] = loaded_images

        preds = model.predict(seq_images)
        output.append(','.join([seq[0]] + [str(num) for num in preds[0]]))

    if len(os.sys.argv) == 1:
        outf = "my_predictions_incep.csv"
    else:
        outf = os.sys.argv[1]

    f = open(outf, 'w')
    f.write('\n'.join(output))
    f.close()

    print("Predictions written to '{}'".format(outf))