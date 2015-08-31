import matplotlib as mpt
mpt.use('Agg')
import matplotlib.pyplot as plt

import skimage.io as sio
import numpy as np

from SRM import SRM

def do_process(img_path):
    """This version will implement the paper version
    """
    im_data = sio.imread(img_path)
    print im_data.shape
    
    srm = SRM(im_data, 256)
    segmented = srm.run()

    # get all edges and get the basic value for each channel
    plt.imshow(segmented/256)
    plt.savefig('tmp.png')


if __name__ == "__main__":
    test_img = '../opium/2012.png'
    do_process(test_img)
