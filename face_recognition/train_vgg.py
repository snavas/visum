import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat
import numpy as np
import os
import re
import glob
import hashlib
import os.path
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.engine import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import pickle
from vggface import define_model
#from image_data_augmenter import ImageDataAugmenter
from utils import *
from data_augmenter import ImageDataAugmenter


MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
MODEL_INPUT_WIDTH = 244
MODEL_INPUT_HEIGHT = 244
MODEL_INPUT_DEPTH = 3


def one_hot_encoder(ids, rows, cols):
    onehot_y = np.zeros((rows, cols))
    for i, item in enumerate(ids):
        onehot_y[i, np.int(item) -1] = 1
    return onehot_y


def create_image_lists(image_dir, testing_percentage, validation_percentage):
  """Builds a list of training images from the file system.

  Analyzes the sub folders in the image directory, splits them into stable
  training, testing, and validation sets, and returns a data structure
  describing the lists of images for each label and their paths.

  Args:
    image_dir: String path to a folder containing subfolders of images.
    testing_percentage: Integer percentage of the images to reserve for tests.
    validation_percentage: Integer percentage of images reserved for validation.

  Returns:
    A dictionary containing an entry for each label subfolder, with images split
    into training, testing, and validation sets within each label.
  """
  if not gfile.Exists(image_dir):
    print("Image directory '" + image_dir + "' not found.")
    return None
  result = {}
  sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
  # The root directory comes first, so skip it.
  is_root_dir = True
  for sub_dir in sub_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    file_list = []
    dir_name = os.path.basename(sub_dir)
    if dir_name == image_dir:
      continue
    print("Looking for images in '" + dir_name + "'")
    for extension in extensions:
      file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
      file_list.extend(gfile.Glob(file_glob))
    if not file_list:
      print('No files found')
      continue
    if len(file_list) < 20:
      print('WARNING: Folder has less than 20 images, which may cause issues.')
    elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
      print('WARNING: Folder {} has more than {} images. Some images will '
            'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    training_images = []
    testing_images = []
    validation_images = []
    for file_name in file_list:
      base_name = os.path.join(os.getcwd(), image_dir, dir_name,os.path.basename(file_name))
      # We want to ignore anything after '_nohash_' in the file name when
      # deciding which set to put an image in, the data set creator has a way of
      # grouping photos that are close variations of each other. For example
      # this is used in the plant disease data set to group multiple pictures of
      # the same leaf.
      hash_name = re.sub(r'_nohash_.*$', '', file_name)
      # This looks a bit magical, but we need to decide whether this file should
      # go into the training, testing, or validation sets, and we want to keep
      # existing files in the same set even if more files are subsequently
      # added.
      # To do that, we need a stable way of deciding based on just the file name
      # itself, so we do a hash of that and then use that to generate a
      # probability value that we use to assign it.
      hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
      percentage_hash = ((int(hash_name_hashed, 16) %
                          (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                         (100.0 / MAX_NUM_IMAGES_PER_CLASS))
      if percentage_hash < validation_percentage:
        validation_images.append(base_name)
      elif percentage_hash < (testing_percentage + validation_percentage):
        testing_images.append(base_name)
      else:
        training_images.append(base_name)
    result[label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
        'validation': validation_images,
    }
  return result


def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)

  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def get_data_lists(data_dict):

    x = np.array([np.array(data_dict[np.str(key)]['training']).flatten() for key in data_dict.keys()])
    x = np.concatenate(x).ravel()
    y = np.array([np.array([key]*len(data_dict[np.str(key)]['training'])).flatten() for key in data_dict.keys()])
    y = np.concatenate(y).ravel()
    assert(len(x) == len(y))
    return (x,y)


if __name__ == "__main__":

    cwd = os.getcwd()

    image_dir = 'data'
    image_dir_test = 'data_test'

    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(image_dir, 0, 0)
    print(image_lists)

    print(image_lists)
    class_count = len(image_lists.keys())
    if class_count == 0:
        print('No valid folders of images found at ' + image_dir)
    if class_count == 1:
        print('Only one valid folder of images found at ' + image_dir +
              ' - multiple classes are needed for classification.')

    train_X, train_Y = get_data_lists(image_lists)
    train_Y = one_hot_encoder(train_Y, len(train_Y), 19)

    # Shuffle
    train_X, train_Y = unison_shuffled_copies(train_X, train_Y)
    #train_X = train_X[:100] #Limit the dataset to test in CPU
    #train_Y = train_Y[:100]

    test_images = [img_path for img_path in glob.glob(os.path.join(os.getcwd(), image_dir_test, "*.jpg"))]

    # Initialize model and params
    np.random.seed(seed=40)

    batch_size = 16
    output_shape = (224, 224, 3)
    n_output = 19
    epochs = 40
    lr = 0.00001
    dropout = 0.3

    # Define model
    model = define_model(n_output, lr, dropout)

    #TODO add data augmenter here

    #TODO create new folder for checkpoints
    # Checkpoint model
    filepath = "weights-improvement-{epoch:03d}-{val_loss:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)

    # Early stopping
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=40, verbose=1, mode='min')

    # List of callbacks to add to fit function
    callbacks_list = [checkpoint, earlystopping]

    augmenter = ImageDataAugmenter(rotation_range=4,
                                   rotation_prob=50,
                                   height_shift_range=0.07,
                                   width_shift_range=0.07,
                                   gaussian=[0, 0.1],
                                   illumination=[-50, 50],
                                   zoom=[0.90, 1.1],
                                   flip=0.5,
                                   #gamma=[0, 0.01],
                                   gamma=None,
                                   contrast=None)
                                   #contrast=None)

    # Fit model and save statistics in hist
    hist = model.fit_generator(
        data_generator(train_X, train_Y, batch_size, output_shape, n_output, augmenter=augmenter, net='vgg'),
        steps_per_epoch=np.ceil(len(train_X) / batch_size), epochs=epochs)

    model.save('algo.hd5')

    #with open('hist_algo.pickle', 'wb') as f:
    #    pickle.dump(hist.history, f)

    # Test
    val_x = [image.img_to_array(image.load_img(fname, target_size=output_shape)) for fname in test_images]
    #val_x = [plt.imread(fname)[..., :3] for fname in test_images]
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

    if len(os.sys.argv) == 1:
        outf="my_predictions_vgg.csv"
    else:
        outf=os.sys.argv[1]

    fids = np.arange(1,len(preds)+1).reshape(len(preds),1)
    preds = np.hstack([fids, preds])

    # write submission file
    fmt='%.1f,' + ','.join(['%.6f']*19)
    np.savetxt(outf, preds, fmt=fmt, header="ID,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19", delimiter=",")

    print("\nPredictions written to '{}'".format(outf))