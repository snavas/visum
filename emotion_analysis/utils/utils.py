# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:18:09 2017

@author: pmmf
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools


def log_loss(y, pred_probs, eps=1e-15):
    N = float(len(y))

    pred_probs[pred_probs <= 0] = eps
    pred_probs[pred_probs >= 1] = 1-eps
    pred_probs = np.log(pred_probs)

    loss = np.sum(np.sum(y * pred_probs, axis=1)) * -1/N
    return loss


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    else:
        print('Confusion matrix, without normalization')
    # print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def imshow(figure, im, cmap=None):
  plt.figure(figure)
  plt.imshow(im,cmap)
  plt.axis("off")
  plt.show()
  plt.draw()
  

def myWHist(M, PDF, args): 
    bins = args[0]
    ranges = args[1]
    (hist, _) = np.histogram(M.ravel(),bins,ranges, weights = PDF.ravel())
    
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return list(hist)

def myHist(M, args): 
    bins = args[0]
    ranges = args[1]
    (hist, _) = np.histogram(M.ravel(),bins,ranges)
    
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return list(hist)

def block_process(M, blk_size, stride, fun=None, args=None):
  rows = []
  out = []
  count = 0
  # rows
  for i in range(0, int(M.shape[0]-stride[0]), int(blk_size[0])):
      cols = []
      # columns
      for j in range(0, int(M.shape[1]-stride[1]), int(blk_size[1])):
          # block indexes
          max_ndx = (min(i+blk_size[0], M.shape[0]), 
                     min(j+blk_size[1], M.shape[1]))
          
          current_block = M[i:int(max_ndx[0]), j:int(max_ndx[1])]  
#          print(">> - ", current_block.shape, i, j)
#          imshow(1, current_block)
          # apply function to current block
          out += fun(current_block, args)
          count+=1
          
          cols.append(current_block)
      rows.append(np.concatenate(cols, axis=1))
  M_new = np.concatenate(rows, axis=0)   
  return M_new, out    


