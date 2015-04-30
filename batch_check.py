"""
1. for each region, get the bounary polygon
2. get the shift of each region, get the real position 
2.1 draw on the figure to find error
3. transform back into latlong coordinate
"""
import matplotlib.pyplot as plt

import sys
import simplejson
import os
import re

import numpy as np
import skimage.io
from pix2latlon import pix2ll_interpolate_warp as pix2ll

t = ''

def batch_show():
    check_list = ['m4-1.png','vh2-1.png','vl13-1.png','vl3-1.png','m2-1.png','vl10-1.png',
            'm6-1.png','vl12-1.png']
    flist = os.listdir('./'+t)
    for fname in flist:
        if not fname in check_list:
            continue
        print fname
        if fname.endswith('png') and fname.count('-1') > 0:
            show( './'+t+'/'+fname )

def show(im_path):

    shift_path = '%s_shift.json'%(t)
    shift_dict = simplejson.load(open(shift_path))
    im_name = im_path[im_path.rindex('/')+1:]
    if t == 'taliban':
        orgim = skimage.io.imread('./taliban/talibancontrol.png')
    elif t == 'opium':
        orgim = skimage.io.imread('./opium/2012-clean.png')
    else:
        raise Exception('no supported')

    # load the polygon data for each region
    poly_path = './shpres/%s.json'%(im_name)
    if not os.path.isfile(poly_path):
        return
    print 'shift', shift_dict[im_name]
    rs, cs = shift_dict[im_name]
    
    pg = simplejson.load(open(poly_path))
    #print pg
    exter = np.array(pg['ext'])
    exter[:,0] += rs
    exter[:,1] += cs
    plt.clf()
    
    plt.subplot(131)
    plt.plot(exter[:,0], exter[:,1])
    
    plt.subplot(132)
    imdata = skimage.io.imread('./mask/%s'%(im_name))
    plt.imshow(imdata)

    plt.subplot(133)
    tmpdata = orgim[ rs:rs+imdata.shape[0], cs:cs+imdata.shape[1], :]
    plt.imshow( tmpdata )
    plt.show()
    #exit()

    #print exter
    inters = pg['intlist']
    print inters
    

if __name__ == "__main__":
    global t
    t = sys.argv[1]
    batch_show()
