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

def read_video(path):
    frames = []
    for img_filename in sorted(list(os.walk(path))[0][2]):
        frames.append(cv2.imread(os.path.join(path, img_filename), 0))
    return np.asarray(frames)


def compute_background_model(frames):
    bg = None
    w = 7

    for img in frames:
        img = cv2.blur(np.copy(img), (w, w))

        if bg is None:
            bg = np.zeros((120, 160))

        bg += img.astype(float)

    return bg / len(frames)


def compute_foreground(frames, bg):
    diff = np.abs(frames.astype(np.float) - bg.astype(np.float))
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff)) * 255.
    thrs, _ = cv2.threshold(diff.astype(np.uint8).ravel(), 0, 255,
                            cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), dtype=np.uint8)
    return np.asarray([cv2.dilate(cv2.erode(b, kernel), kernel)
                        for b in (diff > thrs).astype(np.uint8)])


class SpeedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, scale=1):
        self.scale = scale
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.asarray([self.transform_video(x) for x in X])
    
    def transform_video(self, fg):
        ret = []

        center = [(i, self.get_center(b)) for i, b in enumerate(fg)]
        center = [(i, c) for i, c in center if c is not None]
        if self.scale != 1:
            center = center[::self.scale]

        if len(center) <= 1:
            return [0] * 12
    
        speed = [self.dist(c1, c2, True, True) / float(abs(i2 - i1))
                 for (i1, c1), (i2, c2) in zip(center, center[1:])]
        speed = np.asarray(speed)
        ret += [speed.mean(), speed.max()]

        speed = [self.dist(c1, c2, False, True) / float(abs(i2 - i1))
                 for (i1, c1), (i2, c2) in zip(center, center[1:])]
        speed = np.asarray(speed)
        ret += [speed.mean(), speed.max()]

        speed = [self.dist(c1, c2, True, False) / float(abs(i2 - i1))
                 for (i1, c1), (i2, c2) in zip(center, center[1:])]
        speed = np.asarray(speed)
        ret += [speed.mean(), speed.max()]

        ret += self.center_range(center)

        return ret

    def get_center(self, bimg):
        if np.sum(bimg != 0) < 500:
            return None

        rows = np.repeat(np.arange(bimg.shape[0]),
                        bimg.shape[1]).reshape(bimg.shape)
        cols = np.repeat(np.arange(bimg.shape[1]),
                        bimg.shape[0]).reshape(bimg.shape[::-1]).T
        rows = rows[bimg != 0].mean()
        cols = cols[bimg != 0].mean()

        return rows, cols

    def center_range(self, centers):
        rows = np.asarray([c[0] for _, c in centers])
        cols = np.asarray([c[1] for _, c in centers])
        rows -= rows.mean()
        cols -= cols.mean()
        return [min(rows), max(rows), max(rows) - min(rows),
                min(cols), max(cols), max(cols) - min(cols)]

    def dist(self, (r1, c1), (r2, c2), horizontal=True, vertical=True):
        rdiff = (r2 - r1) ** 2 if vertical else 0.
        cdiff = (c2 - c1) ** 2 if horizontal else 0.
        return math.sqrt(rdiff + cdiff)

class SpatialFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.asarray([self.transform_video(x) for x in X])
    
    def transform_video(self, fg):
        fd, hog_image = hog(np.mean(fg, axis=0),
                            orientations=8,
                            pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), 
			    block_norm='L2-Hys', visualise=True)
        
        avg_fg = np.mean(fg, axis=0)
        avg_fg = avg_fg / np.max(avg_fg) * 255.
        avg_fg = cv2.resize(avg_fg.astype(np.uint8), (5, 5))


        return list(fd) + list(avg_fg.ravel())


#parser.add_argument('--out',type=str,
#      default='predictions.txt',
#      help='File to store predictions (will be overwritten if it exists).'
#  )



f = open('sets/training.csv')
lines = np.asarray(list(csv.reader(f)))
f.close()

print 'Training file loaded.'

#lines = lines[: 100]

videos = lines[:, 0]
labels = lines[:, 1]
    
classes = sorted(list(set(labels)))
features = []

fts_transformer = FeatureUnion([('speed-1', SpeedFeatures(1)),
                                ('speed-2', SpeedFeatures(2)),
                                ('speed-4', SpeedFeatures(4)),
                                ('hog', SpatialFeatures())])

for i, v in enumerate(videos):
    frames = read_video(v)
    bg = compute_background_model(frames)
    fg = compute_foreground(frames, bg)

    ft = fts_transformer.transform([fg])[0]
    features.append(ft)
    #print i + 1, '/', len(videos)
    sys.stdout.write("Compute features, train sample {}/{}                          \r".format(i+1,len(videos)))
    sys.stdout.flush()

print("\ndone.")

print("Start training...   "),
model = RandomForestClassifier(n_estimators=100)
model.fit(features, labels)
print("done.")

f = open('sets/sample_submission.csv')
lines = list(csv.reader(f))
f.close()

output = [','.join(lines[0])]

classes = lines[0][1:]
index_ = [list(model.classes_).index(c) for c in classes]


for i, l in enumerate(lines[1:]):
    frames = read_video(l[0])
    bg = compute_background_model(frames)
    fg = compute_foreground(frames, bg)

    ft = fts_transformer.transform([fg])[0]
    probs = model.predict_proba([ft])[0]
    probs = probs[index_]

    output.append(','.join([l[0]] + map(str, probs)))
    sys.stdout.write("Compute features and inference, test sample {}/{}                          \r".format(i+1,len(lines)-1 ))
    sys.stdout.flush()

print("\nFinished!")

if len(os.sys.argv) == 1:
    outf="my_predictions.csv"
else:
    outf=os.sys.argv[1]

f = open(outf, 'w')
f.write('\n'.join(output))
f.close()

print("Predictions written to '{}'".format(outf))

