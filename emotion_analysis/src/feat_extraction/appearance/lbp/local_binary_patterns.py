import numpy as np
import math
from skimage import feature 
import cv2
import sys
import copy

from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFECV  
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, 'utils/')
from utils import block_process, myHist, imshow, myWHist, log_loss

eps=1e-7

class LocalBinaryPatterns(object):
    def __init__(self, numPoints, radius, method="uniform", multiscale=[1,"im"]):
        # init lb
        self.numPoints = numPoints
        self.radius = radius
        self.method = method
        self.scales = multiscale[0]
        self.scale_type = multiscale[1]
        self.scale_factor = 0.5
        self.hist_bins = np.arange(0, self.numPoints + 3)
        self.hist_range = (0, self.numPoints + 2)
    
    def apply(self, imgs):
        lbps_array = []
        for i, im in enumerate(imgs):  
            lbps = [] 
            image = copy.deepcopy(im)
            numPoints = copy.deepcopy(self.numPoints)
            radius = copy.deepcopy(self.radius)
            method = copy.deepcopy(self.method)
            
            for s in range(0, self.scales):
                # compute lbp
                lbp = feature.local_binary_pattern(image, numPoints, radius, 
                                                   method)
        		                                    
                # scale image or increase lbp npoints/radius
                if self.scale_type == "im":
                    image = cv2.resize(image,None,
                                       fx=self.scale_factor, fy=self.scale_factor, 
                                       interpolation = cv2.INTER_CUBIC)
                else:
                    radius    = radius * (1/self.scale_factor)
                    numPoints = numPoints * (1/self.scale_factor)
                lbps.append(lbp)   
                
            lbps_array.append(lbps)
        return lbps_array
    
    def describe_global(self, imgs_lbps):
        hist_array_imgs = []        
        for i, lbps in enumerate(imgs_lbps): # loop each image
            hist_array = []
            for lbp in lbps:# loop each lbp scale   		                                    
                # compute lbp histogram
                (hist, _) = np.histogram(lbp.ravel(),
                                         self.hist_bins,
                                         self.hist_range)
        		                          
                # normalize the histogram
                hist = hist.astype("float")
                hist /= (hist.sum() + eps)
                
                hist_array.append(hist)
            hist_array = np.array(hist_array).ravel()
            hist_array_imgs.append(hist_array)
        return np.array(hist_array_imgs)   
    
    def describe_local(self, imgs_lbps, grid):
        
        hist_array_imgs = []        
        
        for i, lbps in enumerate(imgs_lbps): # loop each image
            hist_array = []
            args = [self.hist_bins, self.hist_range]
            
            for lbp in lbps:    	
                # compute block size
                blk_size, stride = self.get_BlkStride_size(lbp, grid)
        		   
                # compute lbp local histogram
                (_, hist) = block_process(lbp, blk_size, stride, myHist, args)   

                hist_array.append(hist)
                
            hist_array = np.array(hist_array).ravel()
            hist_array_imgs.append(hist_array)
        return np.array(hist_array_imgs)
    
    def describe_kpts(self, imgs_lbps, kpts, blk_sz):
        # FIXME : resize key-points positions im multiscale
        self.blk_sz = blk_sz
        hist_array_imgs = []   
        for i, lbps in enumerate(imgs_lbps): # loop each image
#            print(i)
            hist_array = []
            args = [self.hist_bins, self.hist_range]
            kpts_x, kpts_y = zip(*kpts[i])
        
                 
            for self.lbp in lbps:
                hist = [myHist(self.get_kpt_blk(x,y), args) 
                                            for (x,y) in zip(kpts_x, kpts_y)]
                hist_array.append(hist)
        
            hist_array = np.array(hist_array).ravel()
            hist_array_imgs.append(hist_array)
        return np.array(hist_array_imgs)
    
    def get_BlkStride_size(self, image, grid):
        blk_height = math.floor(float(image.shape[0]) / float(grid[0]) ) 
        blk_width  = math.floor(float(image.shape[1]) / float(grid[1]) )
        blk_size = (blk_height, blk_width)
        stride = (image.shape[0] - blk_height*grid[0], 
                  image.shape[1] - blk_width*grid[1])
        
        return blk_size, stride
        
    def get_kpt_blk(self, x, y):
        x_ini = max(0, int(x - (self.blk_sz[1] / 2.0)))
        x_end = min(self.lbp.shape[1], int(x + (self.blk_sz[1] / 2.0)))
        
        y_ini = max(0, int(y - (self.blk_sz[0] / 2.0)))
        y_end = min(self.lbp.shape[0], int(y + (self.blk_sz[0] / 2.0)))  
#        imshow(1, self.lbp[y_ini:y_end, x_ini:x_end])
        
        return self.lbp[y_ini:y_end, x_ini:x_end]



    def fit(self, X_train, y_train, cv):
        # model : svm plus data normalization
        clf = svm.SVC(decision_function_shape='ovo', probability=True)

        pipe = make_pipeline(StandardScaler(), 
                             GridSearchCV(clf, param_grid={'C': np.logspace(-3, 2, 30)}, 
                             cv=cv))

        pipe.fit(X_train, y_train)

        return pipe

    def predict(self, model, X_test):
        preds       = model.predict(X_test)
        preds_proba = model.predict_proba(X_test)
        return preds, preds_proba

    def eval(self, preds, preds_proba, Y_test):
            nTest = len(Y_test)
            
            # accuracy
            acc = sum(np.argmax(Y_test, axis=1)==preds)/np.float(nTest)*100

            # logloss
            loss = log_loss(Y_test, preds_proba)

            return acc, loss       