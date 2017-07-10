#!/usr/bin/env python 

import os, sys
import glob
import tensorflow as tf
import numpy as np

# initialize test path
TEST_PATH = 'data_test'
#NET = 'INCEPTION'
test_img_paths = [img_path for img_path in glob.glob(TEST_PATH + "/*.jpg")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]

preds = np.array([]).reshape(0,20)
gt = np.array([]).reshape(0,20)

fid = 1
tot=len(test_img_paths)
for img in sorted(test_img_paths):

    #label = int(img[img.find('00'):img.find('00')+4])
    #c_id = np.zeros(20); c_id[label] = 1    # ground truth for id
    image_data = tf.gfile.FastGFile(img, 'rb').read()

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        # rearrange label order
        aux_predictions = np.zeros(20)
        jj = 0;
        for ii in label_lines:

            aux_predictions[int(ii)] = predictions[0,jj]
            jj += 1

        sys.stdout.write("Inference on image ({}/{}) {}                          \r".format(fid,tot,img))
        sys.stdout.flush()

        # add file id
        aux_predictions[0] = fid
        #c_id[0] = fid
        fid += 1

        # save results and ground truth
        preds = np.vstack([preds, aux_predictions])
        #gt = np.vstack([gt, c_id])


if len(os.sys.argv) == 1:
    outf="my_predictions.csv"
else:
    outf=os.sys.argv[1]


# write submission file
fmt='%.1f,' + ','.join(['%.6f']*19)
np.savetxt(outf, preds, fmt=fmt, header="ID,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16,S17,S18,S19", delimiter=",")

print("\nPredictions written to '{}'".format(outf))
