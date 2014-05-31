import numpy as np
import sklearn as sk
import pandas as pd
from sklearn.decomposition import SparsePCA, PCA

import random

class IterativeSPCA():

    def __init__(self, Npc=None, alpha=1, max_iter=5000, tol=1e-8):
        self.Npc = Npc #number of components
        self.alpha = alpha  # penalty value
        self.max_iter = max_iter # max number of iteration
        self.tol = tol # accepted tollerance error
        

    def spca_iterate(self, u_old, v_old):
        # this is the equivalent of the nipals algorithm but designed in order to 
        # obtain a sparse loading
        v_new = np.zeros(shape=v_old.shape) 
        u_new = np.zeros(shape=u_old.shape)

        for iteration in range(self.max_iter): #repeat the iteration non more than max iter
            y = np.dot(self.X.T, u_old).squeeze()
            v_new = (np.sign(y)*np.maximum(np.abs(y)-self.alpha, 0)).squeeze()
            norm_v = np.linalg.norm(v_new)**2
            if norm_v == 0: #if norm v is 0 it means that no components have been selected. So you should reduce the penalty lambda
                
                break

            x_times_v = np.dot(self.X, v_new)
            x_times_v.shape = (self.m, 1)
            u_new = x_times_v/np.linalg.norm(x_times_v)

            if np.linalg.norm(v_new)==0:#check again if the norm is 0
                break
            if np.linalg.norm(v_new - v_old)<self.tol or np.linalg.norm(-v_new - v_old) < self.tol:#check if there is convergence
                break
            u_old = u_new
            v_old = v_new

        if iteration == self.max_iter-1:
            print("No Convergence. Error!!!")

        norm_v = np.linalg.norm(v_new)
        v_new.shape = (self.n, 1)
        v_final = sk.preprocessing.normalize(v_new, axis=0)
        u_final = u_new * norm_v
        return v_final, u_final

    def fit(self, X):

        self.m, self.n = X.shape
        self.components_ = np.zeros(shape=(self.n,self.Npc))
        self.sT = np.zeros(shape=(self.m,self.Npc))
        self.X = X

        for i in range(self.Npc):
            
            pca = PCA(n_components=1).fit(self.X)# this can be improved using a nipals version of PCA
            v = pca.components_
            v = sk.preprocessing.normalize(v)
            u = self.X.dot(v.T)
            s = np.linalg.norm(u)

            u_old = u
            v_old = v*s

            v_final, u_final = self.spca_iterate(u_old=u_old, v_old=v_old)
            self.sT[:, i] = u_final.squeeze()
            self.components_[:, i] = v_final.squeeze()
            u_final.shape = (self.m, 1)
            self.X = self.X-np.dot(u_final, v_final.T)
        return self.components_, self.sT


import pandas as pd
   
print("start")
for i in range(10):
# Set the dataset 
    X = np.random.random((100, 30))
    X = sk.preprocessing.scale(X)   # scale the data
    alpha = 1 # set the penalty
    nPCs = 4 # set the number of components

    import time
    start_time = time.time()
    a = IterativeSPCA(Npc=nPCs, alpha=alpha)
    sP, sT = a.fit(X[:])
    time1 = time.time()
    spca = SparsePCA(n_components=nPCs, alpha=alpha, ridge_alpha=0)
    spca.fit(X[:])
    time2 = time.time()

     #df = pd.DataFrame(columns=['it_spc1','spc1'])
    #df['it_spc1'] = a.components_[:,0]
    #df['spc1'] = spca.components_.T[:,0]/np.linalg.norm(spca.components_.T[:,0])
    #df['it_spc2'] = a.components_[:,1]
    #df['spc2'] = spca.components_.T[:,1]/np.linalg.norm(spca.components_.T[:,1])


    #print(df)
    print("time iterative SPCA=", time1-start_time, "time SparsePCA=", time2-time1)



print("finish!")
