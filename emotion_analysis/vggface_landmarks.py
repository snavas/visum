import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.engine import Model
from keras_vggface.vggface import VGGFace
from keras.layers import Flatten, Dense, Input, Dropout, Merge
from keras.models import Sequential
from keras import metrics

def define_model(n_output, lr, dropout=1.):
    """
    Define model architecture, optimization and loss function
    """

    hidden_dim = 4096
    image_input = Input(shape=(224, 224, 3), name='image')
    landmarks_input = Input(shape=(136,), name='landmarks')
    weight_init = keras.initializers.glorot_uniform(seed=3)

    #model = Sequential()
    vgg_model = VGGFace(input_tensor=image_input, include_top=True)

    for layer in vgg_model.layers[:4]:
        layer.trainable = False

    #vgg_model.trainable = False
    last_layer = vgg_model.get_layer('fc6/relu').output
    #print(last_layer.output_shape)

    #extra.add(keras.layers.core.Activation('linear', input_shape=(136,)))
    #print(extra.output_shape)

    #x = Merge([last_layer, extra], mode='concat', concat_axis=1)
    x = keras.layers.concatenate([last_layer, landmarks_input])
    x = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init, name='fc7')(x)
    if (dropout < 1.):
        x = Dropout(dropout, seed=0, name='dp7')(x)
    out = Dense(n_output, activation='softmax', kernel_initializer=weight_init, name='fc8')(x)
    #model.add(Merge([last_layer, extra], mode='concat', concat_axis=1))
    #model.add(Dense(hidden_dim, activation='relu', kernel_initializer=weight_init, name='fc7'))
    #model.add(Dense(n_output, activation='softmax', kernel_initializer=weight_init, name='fc8'))

    model = Model(inputs=[image_input, landmarks_input], outputs = out)

    # Load VGG Face model without FC layers
    #vgg_model = VGGFace(input_tensor=image_input, include_top=True)

    # Add new fc layers with random gaussian init
    # weight_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)



    #last_layer = vgg_model.get_layer('fc6/relu').output
    #fc7 = vgg_model.get_layer('fc7')
    #fc7r = vgg_model.get_layer('fc7/relu')
    #x = last_layer



    # x = Flatten(name='flatten')(last_layer)
    # x = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init,  name='fc6')(x)
    #if (dropout < 1.):
    #    x = Dropout(dropout, seed=0, name='dp6')(x)
    #x = fc7(x)
    #x = fc7r(x)

    #x = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init, name='fc7')(x)
    #if (dropout < 1.):
    #    x = Dropout(dropout, seed=1, name='dp7')(x)
    #out = Dense(n_output, activation='softmax', kernel_initializer=weight_init, name='fc8')(x)

    print(len(model.layers))
    print([n.name for n in model.layers])

    #x = vgg_model.get_layer('fc6').output
    #x = Flatten(name='flatten')(last_layer)
    #x = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init, name='fc6')(x)
    #if (dropout < 1.):.
    #    x = Dropout(dropout, seed=0, name='dp6')(x)
    #x = Dense(hidden_dim, activation='relu', kernel_initializer=weight_init, name='fc7')(x)
    #if (dropout < 1.):
    #    x = Dropout(dropout, seed=1, name='dp7')(x)
    #out = Dense(n_output, activation='softmax',  kernel_initializer=weight_init, name='fc8')(x)

    # Freeze first conv layers


    #model = Model(image_input, out)

    # Print model summary
    model.summary()

    # Initialize optimizer
    #https://web.stanford.edu/class/cs231a/prev_projects_2016/CS231AFinal.pdf
    opt = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = keras.optimizers.adam(lr, decay=1e-6)
    #https: // arxiv.org / pdf / 1704.08619.pdf

    # Use mean euclidean distance as loss and angular error and mse as metric
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy', metrics.categorical_accuracy])

    return model

