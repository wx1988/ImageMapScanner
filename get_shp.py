# read the shp file
# draw the polygon

# 1. ignore the small polygon, find the largest polygon
# 2. represent the inner relation

import os
import matplotlib as mpt
mpt.use('Agg')
import matplotlib.pyplot as plt
import shapefile
from shapely.geometry import Polygon, Point
import numpy as np
from find_boundary import get_fg_mask 
import simplejson

def draw_shp_polygon(shp_path):
    sr = shapefile.Reader(shp_path)
    shapes = sr.shapes()

    for i,shape in enumerate(shapes):
        #print shape
        #print shape.points
        #if i != 66 and i!= 41:
        tmp_pts = np.array(shape.points)
        print i, tmp_pts.shape
        plt.clf()
        old_ci = -1
        c_i = 0
        first_poly = True
        exter = None
        inter = []
        for j in range(tmp_pts.shape[0]):
            if j == c_i:
                continue
            #print tmp_pts[j,:], tmp_pts[c_i,:]
            v = tmp_pts[j,:] - tmp_pts[c_i,:]
            d = np.sqrt(np.dot(v,v))
            #print v, d
            if d <= 0.5:
                old_ci = c_i
                c_i = j+1
                plt.plot(tmp_pts[old_ci:c_i,0], tmp_pts[old_ci:c_i,1])
                if first_poly:
                    exter = tmp_pts[old_ci:c_i,:]
                    print exter
                    first_poly = False
                else:
                    tmp_pg = Polygon( tmp_pts[old_ci:c_i,:] )
                    if tmp_pg.area > 4*4:
                        inter.append( tmp_pts[old_ci:c_i,:] )
        pg = Polygon( exter, inter)
        if pg.area < 6*6:
            continue
        for k in range(160,190):
            p = Point(k,150)
            print p, pg.contains(p)
        #plt.show()
        plt.savefig('./shp/tmp/%s-%d.png'%(shp_path[shp_path.rindex('/')+1:],i))

def get_shp_index(shp_path,im_path):
    # get the pos for the good_label
    fg_mask = get_fg_mask(im_path)
    plt.imshow(fg_mask)
    plt.show()
    sr = shapefile.Reader(shp_path)
    shapes = sr.shapes()

    pg_list = []
    max_score = -1
    for i,shape in enumerate(shapes):
        tmp_pts = np.array(shape.points)
        old_ci,c_i,first_poly = -1,0, True
        exter = None
        inter = []
        for j in range(tmp_pts.shape[0]):
            if j == c_i:
                continue
            v = tmp_pts[j,:] - tmp_pts[c_i,:]
            d = np.sqrt(np.dot(v,v))
            if d <= 0.5:
                old_ci = c_i
                c_i = j+1
                plt.plot(tmp_pts[old_ci:c_i,0], tmp_pts[old_ci:c_i,1])
                if first_poly:
                    exter = tmp_pts[old_ci:c_i,:]
                    first_poly = False
                else:
                    # TODO, if area less than 4*4, then remove
                    tmp_pg = Polygon( tmp_pts[old_ci:c_i,:] )
                    if tmp_pg.area > 4*4:
                        inter.append( tmp_pts[old_ci:c_i,:] )
        pg = Polygon( exter, inter)
        if pg.area < 16:
            pg_list.append(None)
            continue
        pg_list.append(pg)
        score = 0
        for ii in range(fg_mask.shape[0]):
            for jj in range(fg_mask.shape[1]):
                if fg_mask[ii,jj] and pg.contains( Point(ii,jj)):
                    score += 1
        print i, score, pg.area
        if score > max_score:
            max_score = score
            mi = i
    print 'max score',max_score, mi
    return mi, pg_list 

def batch():
    flist = os.listdir('./')
    for fname in flist:
        if fname.count('-1') > 0 and \
                fname.count('png') > 0:
            out_path = './shpres/%s.json'%(fname)
            if os.path.isfile(out_path):
                continue
            mi, pg_list = get_shp_index(
                    './shp/%s.shp'%(fname), fname)
            #print mi, len(pg_list)
            pg = pg_list[mi]
            res = {}
            res['ext'] = [list(co) for co in pg.exterior.coords]
            res['intlist'] = []
            for i in pg.interiors:
                res['intlist'].append([list(co) for co in i.coords])
            # write to shpres folder
            outf = open(out_path,'w')
            print>>outf, simplejson.dumps( res )
            outf.close()

if __name__ == "__main__":
    #draw_shp_polygon('./shp/push5-1.png.shp')
    #get_shp_index('./shp/sparse1-1.png.shp', 'sparse1-1.png')
    batch()

