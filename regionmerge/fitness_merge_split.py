"""
This fitnesss test based methods

For one cluster, if there is another split of the region into two continuous region 
with the likelihood larger than exising one,
then split, and recursive checking this. 


For two cluster, if merging them into one will generate better likelihood, 
then merge. 

Use Gaussian assumption. 

"""
import matplotlib as mpt
mpt.use('Agg')
import matplotlib.pyplot as plt

import skimage.io as sio
import numpy as np
from collections import deque

import scipy.spatial.distance
from scipy.stats import multivariate_normal

def do( img_path ):
    im_data = sio.imread( img_path )
    nr, nc, _ = im_data.shape
    print nr, nc 
    print im_data[:,:,0]
    new_data = im_data.reshape( nr*nc , 3)
    print new_data.shape
    print new_data

    #NOTE, there is no parameter at all.
    # first try the T.png

    # fit using one gaussian.
    print np.mean( new_data, axis = 0)
    print np.cov( new_data, rowvar = 0)

    # fit with two gaussian, EM algorithm. 
    # but with limitation that the two separate cluster 
    # TODO, should also be connected in the space 

    # for all the pixels, randomly assign the label as 0 or 1 
    max_iter = 100
    label = np.random.randint(2, size=nr*nc) 
    while True:
        c1_data = new_data[label==0, :]
        mean1 = np.mean( c1_data, axis=0)
        std1 = np.cov( c1_data, rowvar=0)
        #print 'new iteration'
        #print mean1
        #print std1
        std1 = std1 + 0.01*np.max(std1)*np.identity(3)
        
        c2_data = new_data[label==1, :]
        mean2 = np.mean( c2_data, axis=0)
        std2 = np.cov( c2_data, rowvar=0)
        #print mean2
        #print std2
        std2 = std2 + 0.01*np.max(std2)*np.identity(3)

        # NOTE
        p1 = multivariate_normal.pdf( new_data, mean=mean1, cov=std1 )
        p2 = multivariate_normal.pdf( new_data, mean=mean2, cov=std2 )
        new_label = np.zeros( nr*nc )
        new_label[p1<p2] = 1
        #print np.sum( new_label != label)
        if np.sum( new_label != label) < 2:
            label = new_label
            break
        label = new_label
        max_iter -= 1
        if max_iter < 0:
            break

    print max_iter

    # calculate the log likelihood


if __name__ == "__main__":
    do( 'T.png' )
