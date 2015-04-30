import skimage.io as sio
import os

from get_fg_mask import get_fg_mask

#import matplotlib as mpt
#mpt.use('Agg')
import matplotlib.pyplot as plt

import Image

def do_channel():
    folder = './taliban_png'
    flist = os.listdir(folder)
    for fname in flist:
        if fname.endswith('png') and fname.count('-1') > 0:
            if fname.count('hr3-1') == 0:
                continue
            im = sio.imread(folder + '/' + fname)
            #plt.imshow(im)
            #plt.show()
            print fname,im.shape
            im2 = Image.open(folder+'/'+fname)
            print im2.info
            print im2.size, im2.mode
            print  im2.split()
            im3 = im2.convert('RGBA')
            print im3.size, im3.info
            r,g,b,a =  im3.split()
            a.show()
            im3.show()


def do_get_fg_mask():
    folder = './opium'
    #folder = './taliban'
    flist = os.listdir(folder)
    for fname in flist:
        if fname.endswith('png') and fname.count('-1') > 0:
            mask = get_fg_mask(folder+'/'+fname)
            plt.clf()
            plt.imshow(mask)
            plt.savefig('./mask/%s'%fname)


if __name__ == "__main__":
    #do_channel()
    do_get_fg_mask()

