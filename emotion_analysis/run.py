#!/usr/bin/env python

#==============================================================================
# Imports
#==============================================================================
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import sklearn


## Settings from config file
sys.path.insert(0, 'src/')
import config
config.init()

## emotions code
__emotions__ = config.__emotions__

## Import "Feature extraction" functions 
sys.path.insert(0, 'src/feat_extraction/geometry/')
from feat_kpts_geometry import *

sys.path.insert(0, 'src/feat_extraction/appearance/lbp/')
from local_binary_patterns import LocalBinaryPatterns

## Import load_data functions from load_data.py
from load_data import load_csv, load_test_labels



if __name__ == "__main__":
    #==============================================================================
    # Load Dataset
    #==============================================================================
    train_fn = 'data/training.csv'
    test_fn  = 'data/test.csv'

    ## Training data
    X_train, kpts_train, y_train = load_csv(train_fn, istrain=True)
    print(X_train.shape, kpts_train.shape, y_train.shape)

    for i, pic in enumerate(X_train):
        plt.figure()
        plt.imshow(pic,cmap='gray')
        plt.show()
        os.system("pause")

    ## Test data
    X_test, kpts_test = load_csv(test_fn, istrain=False)
    print(X_test.shape, kpts_test.shape)

    ## Load Test Labels
#    test_labels_fn = 'data/test_labels.csv'
#    y_test = load_test_labels(test_labels_fn)
#    print(y_test.shape)


    #==============================================================================
    # Appearance method
    # LBP (global)
    #==============================================================================
    print(">> LBP (global) ------------------------------------------------------")
    numPoints, radius, method, scales = 8,8,"uniform",[1,"lbp"] # lbp features
    lbp      = LocalBinaryPatterns(numPoints, radius, method, scales) # lbp constructor
       

    # train
    n_cv        = 5 # nfolds for cross validation
    lbp_train   = lbp.apply(X_train) # apply lbp
    desc_train  = lbp.describe_global(lbp_train) # histogram of lbp
    model       = lbp.fit(desc_train, np.argmax(y_train, axis=1), n_cv) # train the a model with desc (svm)
    print("TRAIN: Shape of lbp global descriptor: ", desc_train.shape)

    # test
    lbp_test  = lbp.apply(X_test) # apply lbp
    desc_test = lbp.describe_global(lbp_test) # histogram of lbp

    preds, preds_proba = lbp.predict(model, desc_test) # get predictions
#    #acc, log_loss      = lbp.eval(preds, preds_proba, y_test) # evaluation on test set

    print("TEST: Shape of lbp global descriptor: ", desc_test.shape)
#    #print("ACC: ", acc)
#    #print("LOG_LOSS: ", log_loss)
        

    #==============================================================================
    # MAKE A SUBMISSION
    #==============================================================================
    header = [str(idx) + ": " + emo for idx, emo in enumerate(__emotions__)]
    header.insert(0, "image_id")
    output = [','.join(header)]
    for i in range(0, len(X_test)):
        output.append(str(i) + ',' + ','.join(map(str, preds_proba[i])))


    if len(os.sys.argv) == 1:
        outf="my_predictions.csv"
    else:
        outf=os.sys.argv[1]

    f = open(outf, 'w')
    f.write('\n'.join(output))
    f.close()

    print("Predictions written to '{}'".format(outf))

