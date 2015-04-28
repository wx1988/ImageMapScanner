import os
import simplejson

import matplotlib as mpt
mpt.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import find_boundaries
import skimage.io
from sklearn.cluster import KMeans

# Geometry related
import gdal, ogr
import shapefile
from shapely.geometry import Polygon, Point

from get_fg_mask import get_fg_mask

def do_one_shp(im_name):
    """
    given the image find and get the polygon
    """
    #http://gis.stackexchange.com/questions/78023/gdal-polygonize-lines

    # create mask image here, 
    # generate the shp based on these data
    im_path = './rawdata/%s'%(im_name)
    mask = get_fg_mask(im_path)
    mask_img = np.ones_like(mask).astype(np.float32)
    mask_img[mask] = 0
    skimage.io.imsave('./mask/%s'%(im_name), mask_img)
    tmpdatapath = './mask/%s'%(im_name)
    raster = gdal.Open(tmpdatapath)
    band = raster.GetRasterBand(1)
    
    outname = './shp/%s.shp'%(im_name)
    shp_drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outname):
        shp_drv.DeleteDataSource(outname)
    out_source = shp_drv.CreateDataSource(outname)
    out_layer = out_source.CreateLayer(im_name, geom_type=ogr.wkbLineString )
    gdal.Polygonize(band, None, out_layer, 1)

def debug_show_shp(fname):
    shp_path = './shp/%s.shp'%fname
    sr = shapefile.Reader(shp_path)
    shapes = sr.shapes()

    for i,shape in enumerate(shapes):
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

def get_shp_index(im_name,debug=False):
    shp_path = './shp/%s.shp'%(im_name)
    im_path = './rawdata/%s'%(im_name)
    # get the pos for the good_label
    fg_mask = get_fg_mask(im_path)
    if debug:
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
        print fg_mask.shape[0], fg_mask.shape[1]
        for ii in range(fg_mask.shape[0]):
            for jj in range(fg_mask.shape[1]):
                # NOTE, the Point with first dimension as col, second dimension as row
                if fg_mask[ii,jj] and pg.contains( Point(jj,ii)):
                    score += 1
        print i, score, pg.area
        if score > max_score:
            max_score = score
            mi = i
    print 'max score',max_score, mi
    return mi, pg_list 

def batch_find_json():
    flist = os.listdir('./rawdata')
    for fname in flist:
        if fname.count('-1') > 0 and \
                fname.count('png') > 0:
            out_path = './shpres/%s.json'%(fname)
            if os.path.isfile(out_path):
                continue
            mi, pg_list = get_shp_index(fname)
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
    debug = True
    debug = False
    if debug:
        #im_name = "aimak1-1.png"
        #im_name = "sparse1-1.png"
        fname = 'push11-1.png'
        #do_one_shp(fname)
        #debug_show_shp(fname)
        #get_shp_index(fname,True)
        exit()
    else:
        batch_find_json()
        exit()
        flist = os.listdir('./rawdata')
        for fname in flist:
            if fname.count('-1') > 0 and \
                    fname.count('png') > 0:
                do_one_shp(fname)
