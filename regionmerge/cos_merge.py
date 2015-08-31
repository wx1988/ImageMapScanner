import matplotlib as mpt
mpt.use('Agg')
import matplotlib.pyplot as plt

import skimage.io as sio
import numpy as np
from collections import deque

import scipy.spatial.distance

# NOTE, the cosine is not woking because of the 
# grey level problem.
threshold = 0.9999

"""
The cosine is only on the correlation between the channels
The mean value is one good discriminator, 
and covariance for the relation between channels 
"""

def do_merge( img_path ):
    im_data = sio.imread( img_path )
    nr, nc, _ = im_data.shape

    mask = np.zeros( (nr,nc) )
    cur_cluster = 0

    for r in range(nr):
        for c in range(nc):
            if mask[r,c] > 0:# already added to other clusters                
                continue
            # check 8 directions of neighbour
            cur_cluster += 1
            mask[r,c] = cur_cluster

            # use queue to add other pixels
            tmp_queue = deque()
            tmp_queue.append( [r,c] )
            while len(tmp_queue) > 0:
                tr,tc = tmp_queue.popleft()
                # check 8 directions
                nlist = []
                for rd in [-1,0,1]:
                    for cd in [-1,0,1]:
                        if rd == 0 and cd == 0:
                            continue
                        new_r = rd+tr
                        new_c = cd+tc
                        if new_r < 0 or new_r >= nr:
                            continue
                        if new_c < 0 or new_c >= nc:
                            continue
                        if mask[new_r,new_c] > 0:
                            continue
                        nlist.append( [new_r, new_c] )
                for ttr,ttc in nlist:
                    # add to current cluster only if similar to [tr,tc]
                    cv = scipy.spatial.distance.cosine(im_data[tr,tc,:], im_data[ttr,ttc,:]) 
                    #print tr,tc,im_data[tr,tc,:], im_data[ttr,ttc,:],cv
                    if cv < 1- threshold:
                        mask[ttr,ttc] = cur_cluster
                        tmp_queue.append( [ttr,ttc] )

                    # else do_nothing
    plt.imshow( mask )
    plt.savefig('back_char_seg.png')
                    
if __name__ == "__main__":
    test_img = 'back_char.png'
    do_merge(test_img)
