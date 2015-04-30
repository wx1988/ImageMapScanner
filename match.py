debug = False
#debug = True


import os
import matplotlib as mpt
if not debug:
    mpt.use('Agg')
import matplotlib.pyplot as plt
import skimage.io
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
import simplejson

from get_fg_mask import get_fg_mask_data


def get_mvn(s_im_data):
    r_s_im_data = np.reshape(
            s_im_data, 
            (s_im_data.shape[0]* s_im_data.shape[1], s_im_data.shape[2]))
    mask = get_fg_mask_data(s_im_data) 
    if debug:
        plt.imshow(mask)
        plt.show()
    mask_ravel = np.reshape( mask, (s_im_data.shape[0]*s_im_data.shape[1],1))
    mask_data = r_s_im_data[np.where(mask_ravel)[0],:]
    #print mask_data.shape, r_s_im_data.shape

    mu = np.mean(mask_data ,axis=0)
    n_mask_data = mask_data -mu 
    sigma = np.dot(n_mask_data.T, n_mask_data) / mask_data.shape[0]

    #print n_mask_data
    #print np.mean(n_mask_data, axis=0)
    print mu, sigma
    return mu, sigma, mask

def gen_prob(s_im_info, b_im_data):
    mu, sigma, _ = s_im_info
    r_b_im_data = np.reshape( 
            b_im_data, 
            (b_im_data.shape[0]* b_im_data.shape[1], b_im_data.shape[2]))
    tmp_prob = multivariate_normal.pdf( r_b_im_data, mean=mu, cov=sigma)
    prob_mask = np.reshape( tmp_prob, (b_im_data.shape[0], b_im_data.shape[1]))
    return prob_mask

def match_score(s_im_info, l_im_data):
    # TODO, abandoned
    # TODO, just calculate the prob once
    # No repeat calculation.
    """
    get the colored part distribution, 
    For the colored part, the corresponding l_im_data should belong to this distribution,
    For the uncolored part, the corresponding l_im_data should not belong to this distribution.
    """
    mu, sigma, mask = s_im_info
    r_mask = np.reshape(mask, (mask.shape[0]*mask.shape[1],1))
    # get the probability
    r_l_im_data = np.reshape(
            l_im_data, 
            (l_im_data.shape[0]*l_im_data.shape[1],l_im_data.shape[2]))
    prob_list = multivariate_normal.pdf(r_l_im_data, mean=mu, cov = sigma)
    #print 'old prob',prob_list, prob_list.shape
    #print 'r_mask',r_mask[:,0],r_mask.shape
    ri,_ = np.where(r_mask==False)
    prob_list[ri] = 1 - prob_list[ri]
    #print 'reverse prob', prob_list
    return sum(np.log(prob_list))

    
def do_match(s_im_path, b_im_path,cache_folder='./cache'):
    """
    s_im_path, the full path for small image
    b_im_path, the full path for large image
    """
    print s_im_path
    
    # Some improvement could be made, 
    # 1. use GPU to speed up
    # 2. enlarge the image to include more boundary
    # read data

    if s_im_path.count( '/' ) > 0:
        s_im_name = s_im_path[s_im_path.rindex('/')+1:]
    else:
        s_im_name = s_im_path
    s_im_data = skimage.io.imread(s_im_path)

    # TODo, for the transparent part
    #print s_im_data.shape
    #print s_im_data[:,:,3]
    #exit()

    s_im_data = s_im_data[:,:,0:3]
    if debug:
        plt.imshow(s_im_data)
        plt.show()

    # add ten pixel to each dimension, to enhance the boundary condition
    shift = 10
    tmp_s_im_data = 255*np.ones( (s_im_data.shape[0]+2*shift, s_im_data.shape[1]+2*shift,3))
    tmp_s_im_data[shift:shift+s_im_data.shape[0], shift:shift+s_im_data.shape[1], :] = s_im_data
    s_im_data = tmp_s_im_data

    # basic information of the templates
    sh,sw,_ = s_im_data.shape
    mu, sigma, mask = get_mvn(s_im_data)

    b_im_data = skimage.io.imread(b_im_path)
    b_im_data = b_im_data[:,:,0:3]

    prob_mask = gen_prob( (mu,sigma, mask), b_im_data)
    log_prob_mask = np.log(prob_mask)
    minv = np.nanmin( log_prob_mask[np.logical_not(np.isinf(log_prob_mask))] )
    log_prob_mask[np.isinf(log_prob_mask)] = minv 
    #np.finfo(np.float64).min*10e-6

    max_pdf = multivariate_normal.pdf( mu, mean=mu, cov=sigma)
    log_minus1_prob_mask = np.log(max_pdf-prob_mask)
    mminv = np.nanmin(log_minus1_prob_mask[np.logical_not(np.isinf(log_minus1_prob_mask))])
    log_minus1_prob_mask[np.isinf(log_minus1_prob_mask)] = mminv 
    #np.finfo(np.float64).min *10e-6

    # TODO, change inf to the smallest value

    #if True:
    #if False:
    if debug:
        print 'max pdf', max_pdf, 'nanmin', np.nanmin(log_prob_mask)
        print log_prob_mask
        print np.sum(np.isinf(log_prob_mask))
        print log_minus1_prob_mask
        print np.sum(np.isinf(log_minus1_prob_mask))

        plt.subplot(121)
        plt.imshow(log_prob_mask)
        plt.subplot(122)
        plt.imshow(log_minus1_prob_mask)
        plt.show()
        #return

    # iterate through all possible
    cache_path = '%s/%s_%d.npy'%(
            cache_folder,s_im_name,shift)
    if os.path.isfile(cache_path):
        score = np.load(cache_path)
        #print score.shape
        score_min = np.min(score)
        score[ b_im_data.shape[0]-sh:,:] = score_min#np.finfo('d').min
        score[ :, b_im_data.shape[1]-sw:] = score_min#np.finfo('d').min
    else:
        #score = np.finfo('d').min*np.ones( (b_im_data.shape[0], b_im_data.shape[1] ))
        score = -1000000.*np.ones( (b_im_data.shape[0], b_im_data.shape[1] ))
        for i in range(b_im_data.shape[0]):
            #print 'working on ',i
            # This is too slow, about 1 second per row
            # How to speed up this?
            if i + sh > b_im_data.shape[0]:
                break
            for j in range(b_im_data.shape[1]):
                if j+sw > b_im_data.shape[1]:
                    break
                # TODO, use the prob already generated. 
                tmp_log_prob_mask = log_prob_mask[i:i+sh, j:j+sw]
                tmp_log_minus1_prob_mask = log_minus1_prob_mask[i:i+sh, j:j+sw]
                s = np.sum( tmp_log_prob_mask[mask] )
                s += np.sum( tmp_log_minus1_prob_mask[mask==False] )
                score[i,j] = s

        score_min = np.min(score)
        score[ b_im_data.shape[0]-sh:,:] = score_min#np.finfo('d').min
        score[ :, b_im_data.shape[1]-sw:] = score_min#np.finfo('d').min

        np.save(cache_path, score)
    pos = np.argmax(score)

    r = pos / score.shape[1] + shift
    c = pos % score.shape[1] + shift
    print s_im_path, r, c
    if debug:
        plt.imshow(score)
        plt.show()
    return r,c

def batch(folder, full_im_name, cache_folder,out_path,rev=False):
    """
    folder, the folder hold all image files
    full_im_name, the image to be matched
    """
    flist = os.listdir(folder)
    fname2shift = {}
    #for fname in flist:
    if rev:
        flist = reversed(flist)
    for fname in flist:
        if fname.count('-1') > 0 and \
                fname.count('png') > 0:
            #and \
            #not os.path.isfile('./cache/%s_10.npy'%(fname)):

            # TODO, make this folder consistent
            r,c = do_match(
                    folder+'/'+fname,
                    folder+'/'+full_im_name,
                    cache_folder)
            fname2shift[fname] = [int(r),int(c)]

    outf = open(out_path,'w')
    print>>outf, simplejson.dumps( fname2shift )
    outf.close()
 

if __name__ == "__main__":
    #do_match('aimak5-1.png','afghanistan_ethnoling_97.png')
    #do_match('aimak1-1.png','afghanistan_ethnoling_97.png')
    #do_match('aimak2-1.png','afghanistan_ethnoling_97.png')
    #do_match('aimak3-1.png','afghanistan_ethnoling_97.png')
    #do_match('aimak4-1.png','afghanistan_ethnoling_97.png')
    #bid = 3
    #do_match('baloch%d-1.png'%(bid),'afghanistan_ethnoling_97.png')
    #do_match('sparse1-1.png','afghanistan_ethnoling_97.png')
    batch('./opium','2012-clean.png','./opium_cache','opium_shift.json',True) 
    #batch('./taliban','talibancontrol.png','./taliban_cache','taliban_shift.json') 
    #do_match('./taliban/mc1-1.png','./taliban/talibancontrol.png','./taliban_cache')
    #do_match('./taliban/hr3-1.png','./taliban/talibancontrol.png','./taliban_cache')
   
