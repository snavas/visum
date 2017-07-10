#==============================================================================
# Imports
#==============================================================================
import sys
import os
import numpy as np
import matplotlib.pyplot as plt


## Settings from config file
sys.path.insert(0, 'config/')
import config
config.init()

## emotions code
__emotions__ = config.__emotions__


#==============================================================================
# READ TEST labels
#==============================================================================
def load_test_labels(data_fn):
    import pandas as pd
    from pandas.io.parsers import read_csv

    df = read_csv(os.path.expanduser(data_fn))    
    # y
    Y = df[df.columns[1:]].values
    Y = Y.astype(np.int32)
    return Y

#==============================================================================
# READ TRAINING and TEST DATA 
#==============================================================================
def load_csv(data_fn, istrain=True):
    import pandas as pd
    from pandas.io.parsers import read_csv

    df = read_csv(os.path.expanduser(data_fn))
    # images
    df['image'] = df['image'].apply(lambda im: np.fromstring(im, sep=' '))
    X = np.vstack(df['image'].values)
    X = X.reshape(-1, 238, 244)
    X = X.astype(np.int32)

    # kpts_x
    df['kpts_x'] = df['kpts_x'].apply(lambda im: np.fromstring(im, sep=' '))
    k_x = np.vstack(df['kpts_x'].values)
    k_x = k_x.astype(np.float32)

    # kpts_y
    df['kpts_y'] = df['kpts_y'].apply(lambda im: np.fromstring(im, sep=' '))
    k_y = np.vstack(df['kpts_y'].values)
    k_y = k_y.astype(np.float32)

    # kpts
    kpts = np.stack((k_x, k_y), axis=2)

    if istrain:
        # y
        df[df.columns[-1]] = df[df.columns[-1]].apply(lambda im: np.fromstring(im, sep=' '))
        Y = np.vstack(df[df.columns[-1]].values)
        Y = Y.astype(np.int32)
        return X, kpts, Y
    else:
        return X, kpts


if __name__ == "__main__":
    #==============================================================================
    # Load Dataset
    #==============================================================================
    train_fn = 'data/training.csv'
    test_fn  = 'data/test.csv'

    ## Training data
    X_train, kpts_train, y_train = load_csv(train_fn, istrain=True)
    print(X_train.shape, kpts_train.shape, y_train.shape)

    ## Test data
    X_test, kpts_test = load_csv(test_fn, istrain=False)
    print(X_test.shape, kpts_test.shape)

    #==============================================================================
    # DISPLAY 
    #==============================================================================
    # display tranining sample
    idx = 0

    y_labels = np.argmax(y_train, axis=1)
    x, y = zip(*kpts_train[0])       
    plt.figure()
    plt.clf()
    plt.imshow(X_train[idx],'gray')
    plt.plot(x, y, 'go')
    plt.text(20, 50, __emotions__[y_labels[idx]], style='italic', bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    plt.axis("off")
    plt.show()    