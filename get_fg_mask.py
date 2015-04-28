import skimage.io
import numpy as np
from sklearn.cluster import KMeans

def get_fg_mask(im_path):
    im_data = skimage.io.imread(im_path)
    im_data = im_data[:,:,0:3]
    #print im_data.shape
    r_im_data = np.reshape( im_data, (im_data.shape[0]*im_data.shape[1], im_data.shape[2]))
    km = KMeans(n_clusters=2)
    km.fit(r_im_data)
    mu0 = np.mean( r_im_data[km.labels_ == 0], axis=0)
    mu1 = np.mean( r_im_data[km.labels_ == 1], axis=0)
    print mu0,mu1
    if sum(mu0) < 30:
        good_label = 1
        bg_label = 0
    else:
        good_label = 0
        bg_label = 1

    mask_data = np.reshape(km.labels_==good_label, (im_data.shape[0], im_data.shape[1]) )
    return mask_data


