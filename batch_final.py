"""
1. for each region, get the bounary polygon
2. get the shift of each region, get the real position 
2.1 draw on the figure to find error
3. transform back into latlong coordinate
"""
import matplotlib.pyplot as plt

from get_shp import get_shp_index
import simplejson
import os
import numpy as np
import skimage.io

shift_path = 'shift.json'
shift_dict = simplejson.load(open(shift_path))

def do(im_name):
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
    plt.subplot(121)
    plt.plot(exter[:,0], exter[:,1])
    plt.subplot(122)
    imdata = skimage.io.imread('./mask/%s'%(im_name))
    plt.imshow(imdata)
    plt.show()
    #print exter
    inters = pg['intlist']
    print inters
    
    # get the shift for each region
    
    # tranform the pixel position into lat/long

if __name__ == "__main__":
    flist = os.listdir('./')
    for fname in flist: 
        if fname.count('-1') > 0 and fname.count('png') > 0:
            print fname
            do(fname)
