#import matplotlib.pyplot as plt
#import skimage.io

import simplejson
import os
import re

import numpy as np
from pix2latlon import pix2ll_interpolate_warp as pix2ll
from shapely.geometry import Polygon, Point

#t = 'opium'
t = 'taliban'
shift_path = '%s_shift.json'%(t)
shift_dict = simplejson.load(open(shift_path))

def pg_pix2latlon_strdf(im_name):
    # load the polygon data for each region
    poly_path = './shpres/%s.json'%(im_name)
    if not os.path.isfile(poly_path):
        #print 'no shape files', im_name
        exit()

    #print 'shift', shift_dict[im_name]
    rs, cs = shift_dict[im_name]

    pg = simplejson.load(open(poly_path))
    #print pg

    exter = np.array(pg['ext'])
    exter[:,0] += cs
    exter[:,1] += rs
    ex_lonlat = pix2ll(exter,t)

    inters = pg['intlist']
    
    inter_list = []
    for inter in inters:
        np_inter = np.array(inter)
        np_inter[:,0] += cs
        np_inter[:,1] += rs
        np_inter = pix2ll(np_inter,t)
        inter_list.append(np_inter)
    
    pg_obj = Polygon(ex_lonlat, inter_list)
    print pg_obj.contains( Point(69.178746, 35.813774) )
    #print pg_obj

def export_n3_full(out_path, predicate, folder):
    flist = os.listdir(folder)
    for fname in flist: 
        if fname.count('-1') > 0 and fname.count('png') > 0:
            #print fname
            pg_str = pg_pix2latlon_strdf(fname) 
           
if __name__ == "__main__":
    #pg_pix2latlon_strdf('push5-1.png')
    #pg_pix2latlon_strdf('tajik11-1.png')
    #exit()
    #export_n3()
    #export_n3_full( 'opium.nt', 'opiumprod', './opium')
    export_n3_full( 'taliban.nt', 'talibancontrol', './taliban')
    #show( 'taliban/hr3-1.png') 
    #
