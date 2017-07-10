def init():
    # emotions code
    global __emotions__
    __emotions__ = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

    # neural nets
    global __HEIGHT__, __WITDH__, __DATA_SIZE__, __PRIOR__    
    global __EPOCHS__, __BATCH_SIZE__, __TRAIN_SIZE__, __TEST_SIZE__

    __HEIGHT__     = 90 #238
    __WITDH__      = 90 #244
    __DATA_SIZE__  = (None, 1, __HEIGHT__, __WITDH__)
    __EPOCHS__     = 500
    __BATCH_SIZE__ = 50
    __TRAIN_SIZE__ = 0.75  # for fitting :: val_size = 1-__TRAIN_SIZE__

if __name__ == "__main__":
    init()