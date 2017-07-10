import os
import numpy as np
import math


from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFECV  
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import sys

sys.path.insert(0, 'utils/')
from utils import log_loss

class feat_kpts_geometry(object):
    def __init__(self):
       self.rel_x = []
       self.rel_y = []
       self.dist = []
       self.angle = []
      
    def describe(self, kpts):
        kpts_desc_array = []
        for k in kpts:
            # initialize lists
            self.rel_x = []
            self.rel_y = []
            self.dist = []
            self.angle = []
            
            kpts_y, kpts_x = zip(*k)

            # face central point
            xmean = np.mean(kpts_x)
            ymean = np.mean(kpts_y)
            
            # distance between each kpt and central point in both axes
            xcentral = [(x-xmean) for x in kpts_x]
            ycentral = [(y-ymean) for y in kpts_y]
            
    #        print(xmean, ymean)
    #        print(xcentral, ycentral)
            
            # anglenose
            if kpts_x[26] == kpts_x[29]:
                anglenose = 0
            else:
                anglenose = int(math.atan((kpts_y[26]-kpts_y[29])/
                                          (kpts_x[26]-kpts_x[29]))
                                                   *180.0/math.pi)
            
            if anglenose < 0:
                anglenose += 90
            else:
                anglenose -= 90
            
    #        print(anglenose)
            # landmarks vectorised
            for x, y, w, z in zip(xcentral, ycentral, kpts_x, kpts_y):
                # relative coords
                self.rel_x.append(x) # FIXME :: Normalize
                self.rel_y.append(y)
                
    #            print(self.rel_x, x)
                
                # euclidean distance
                meannp = np.asarray((ymean, xmean))
                coornp = np.asarray((z, w))
                
                dist = np.linalg.norm(coornp-meannp)
                self.dist.append(dist)
                
                # angle relative
                angle_rel = (math.atan((z-ymean)/(w-xmean))*180.0/math.pi) \
                            - anglenose
                
                self.angle.append(angle_rel)
    
            n_kpts = np.array(self.rel_x).shape[0]
            kpts_desc = (np.concatenate(
                    (
                     np.array(self.rel_x).reshape((1, n_kpts)),
                     np.array(self.rel_y).reshape((1, n_kpts)),
                     np.array(self.dist).reshape((1, n_kpts)),
                     np.array(self.angle).reshape((1, n_kpts))
                     ), 
                                        axis=1)
                    )

            kpts_desc_array.append(kpts_desc.ravel())           
        return np.array(kpts_desc_array)
    

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
        
        
        
