import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras.layers import Flatten, Dense, Input, Dropout
from keras import metrics

from keras.applications.resnet50 import ResNet50
from keras import backend as K

def define_model(n_output, lr, dropout=1.):
    """
    Define model architecture, optimization and loss function
    """
    """""
    image_input = Input(shape=(None, 224, 224, 3))

    convnet = ResNet50(weights='imagenet', include_top=False)

    weight_init = keras.initializers.glorot_uniform(seed=3)

    #x = model.get_layer('flatten_1').output
    #out = Dense(n_output, activation='softmax', kernel_initializer=weight_init, name='fc8')(x)

    #for layer in model.layers[:-1]:
    #    layer.trainable = False
    convnet.trainable = False
    encoded_frame = keras.layers.TimeDistributed(convnet)(image_input)

    #encoded_frame2 = keras.layers.core.Reshape((None, None, 2048))(encoded_frame)
    sq_encoded_frame = K.squeeze(K.squeeze(encoded_frame, axis=2),axis=2)

    encoded_vid = keras.layers.LSTM(256)(sq_encoded_frame)
    out = Dense(n_output, activation='softmax', kernel_initializer=weight_init, name='fc8')(encoded_vid)
    """""
    video = keras.layers.Input(shape=(None, 200, 200, 3), name='video_input')

    K.set_learning_phase(0) # TODO what is this?

    cnn = keras.applications.InceptionV3(weights='imagenet',
                                         include_top='False',
                                         pooling='avg')

    weight_init = keras.initializers.glorot_uniform(seed=3)

    cnn.trainable = False
    encoded_frame = keras.layers.TimeDistributed(keras.layers.Lambda(lambda x: cnn(x)))(video)
    encoded_vid = keras.layers.LSTM(256)(encoded_frame)
    outputs = keras.layers.Dense(n_output, activation='softmax', kernel_initializer=weight_init, name='fc8')(encoded_vid)

    model = Model(inputs=[video], outputs=outputs)

    model.summary()

    print(len(model.layers))
    print([n.name for n in model.layers])

    # Initialize optimizer
    # opt = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = keras.optimizers.adam(lr, decay=1e-6)
    opt = keras.optimizers.adam()

    # Use mean euclidean distance as loss and angular error and mse as metric
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=[metrics.categorical_accuracy])


    return model
