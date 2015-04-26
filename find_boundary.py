import os

from skimage.segmentation import find_boundaries
import skimage.io

import gdal, ogr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def do_one(im_path):
    im_data = skimage.io.imread(im_path)
    im_data = im_data[:,:,0:3]
    #print im_data.shape
    r_im_data = np.reshape( im_data, (im_data.shape[0]*im_data.shape[1], im_data.shape[2]))
    km = KMeans(n_clusters=2)
    km.fit(r_im_data)
    mu0 = np.mean( r_im_data[km.labels_ == 0], axis=0)
    mu1 = np.mean( r_im_data[km.labels_ == 1], axis=0)
    if sum(mu0) < 10:
        good_label = 1
        bg_label = 0
    else:
        good_label = 0
        bg_label = 1

    mask_data = np.reshape(km.labels_, (im_data.shape[0], im_data.shape[1]) )
    bd = find_boundaries(
            mask_data, mode='inner', background=bg_label)\
                    .astype(np.uint8)

    plt.subplot(121)
    plt.imshow(mask_data)
    plt.subplot(122)
    plt.imshow(bd)
    tmpdata = np.ones_like(mask_data).astype(np.float32)
    tmpdata[mask_data == bg_label] = 0
    skimage.io.imsave( './shp/%s'%(im_path), tmpdata)
    #plt.show()
    
    tmpdatapath = './shp/%s'%(im_path)
    raster = gdal.Open(tmpdatapath)
    band = raster.GetRasterBand(1)
    
    outname = './shp/%s.shp'%(im_path)
    shp_drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outname):
        shp_drv.DeleteDataSource(outname)
    out_source = shp_drv.CreateDataSource(outname)
    out_layer = out_source.CreateLayer(im_path, geom_type=ogr.wkbLineString )

    # TODO, what is the data type here?
    gdal.Polygonize(band, None, out_layer, 1)

def do_polygon(im_path):
    #http://www.gdal.org/index.html
    #http://gis.stackexchange.com/questions/78023/gdal-polygonize-lines

    pass


if __name__ == "__main__":
    #im_path = "aimak1-1.png"
    im_path = "sparse1-1.png"
    do_one(im_path)
    #do_polygon(im_path)
