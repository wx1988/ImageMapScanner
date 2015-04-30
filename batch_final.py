"""
1. for each region, get the bounary polygon
2. get the shift of each region, get the real position 
2.1 draw on the figure to find error
3. transform back into latlong coordinate
"""
import matplotlib.pyplot as plt

import simplejson
import os
import re

import numpy as np
import skimage.io
from pix2latlon import pix2ll_interpolate_warp as pix2ll

t = 'opium'
#t = 'taliban'
shift_path = '%s_shift.json'%(t)
shift_dict = simplejson.load(open(shift_path))

def pg_pix2latlon_strdf(im_name):
    # load the polygon data for each region
    poly_path = './shpres/%s.json'%(im_name)
    if not os.path.isfile(poly_path):
        print 'no shape files', im_name
        exit()

    print 'shift', shift_dict[im_name]
    rs, cs = shift_dict[im_name]

    pg = simplejson.load(open(poly_path))
    #print pg
    exter = np.array(pg['ext'])
    exter[:,0] += cs
    exter[:,1] += rs
    ex_lonlat = pix2ll(exter,t)
    ex_str = ','.join(['%f %f'%(ell[0],ell[1]) for ell in ex_lonlat])
    print np.min(ex_lonlat,axis=0)
    print np.max(ex_lonlat,axis=0)
    #print ex_str
    #print exter
    inters = pg['intlist']
    #print inters
    in_str_list = []
    for inter in inters:
        np_inter = np.array(inter)
        np_inter[:,0] += cs
        np_inter[:,1] += rs
        np_inter = pix2ll(np_inter,t)
        in_str = ','.join(['%f %f'%(ill[0],ill[1]) for ill in np_inter])
        in_str_list.append(in_str)
    p_str = 'POLYGON('
    p_str += '(%s)'%(ex_str)
    if len(in_str_list) > 0:
        for in_str in in_str_list:
            p_str += ',(%s)'%(in_str)
    p_str += ')'
    #print p_str
    return p_str

def export_n3_full(out_path, predicate, folder):

    #out_path = 'ethno.nt'
    outf = open(out_path,'w')
    print>>outf,'@prefix\tmech:\t<http://students.cse.tamu.edu/xingwang/semanticweb/mech#>.'
    print>>outf,'@prefix\tstrdf:\t<http://strdf.di.uoa.gr/ontology#>.'
    # gen data here 
    flist = os.listdir(folder)
    for fname in flist: 
        if fname.count('-1') > 0 and fname.count('png') > 0:
            print fname
            pg_str = pg_pix2latlon_strdf(fname) 
            tmp_str = 'mech:%s\n'%(fname[:fname.index('.')].replace('-','_'))
            m = re.search('[a-zA-Z]+',fname)
            tmp_str += '\tmech:%s\tmech:%s;\n'%(predicate, m.group())
            tmp_str += '\tstrdf:hasGeometry\t"%s"^^strdf:WKT.'%(pg_str)
            print>>outf,tmp_str
    outf.close()

if __name__ == "__main__":
    #pg_pix2latlon_strdf('push5-1.png')
    #pg_pix2latlon_strdf('tajik11-1.png')
    #exit()
    #export_n3()
    export_n3_full( 'opium.nt', 'opiumprod', './opium')
    #export_n3_full( 'taliban.nt', 'talibancontrol', './taliban')
    #show( 'taliban/hr3-1.png') 
    #batch_show()
