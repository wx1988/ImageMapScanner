import os
import matplotlib.pyplot as plt
import skimage.io
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

def get_mvn(s_im_data):
    r_s_im_data = np.reshape(
            s_im_data, 
            (s_im_data.shape[0]* s_im_data.shape[1], s_im_data.shape[2]))

    km = KMeans(n_clusters=2)
    km.fit(r_s_im_data)
    mu0 = np.mean( r_s_im_data[km.labels_ == 0], axis=0)
    mu1 = np.mean( r_s_im_data[km.labels_ == 1], axis=0)
    if sum(mu0) < 10:
        good_label = 1
    else:
        good_label = 0
    mask_data = r_s_im_data[ km.labels_ == good_label]
    
    mask = np.reshape(
            km.labels_ == good_label, 
            (s_im_data.shape[0], s_im_data.shape[1]))
    #plt.imshow(mask)
    #plt.show()

    mu = np.mean(mask_data ,axis=0)
    n_mask_data = mask_data -mu 
    sigma = np.dot(n_mask_data.T, n_mask_data) / mask_data.shape[0]

    #print n_mask_data
    #print np.mean(n_mask_data, axis=0)
    print sigma
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

    
def do_match(s_im_path, b_im_path):
    # Some improvement could be made, 
    # 1. use GPU to speed up
    # 2. enlarge the image to include more boundary
    # read data
    s_im_data = skimage.io.imread(s_im_path)
    s_im_data = s_im_data[:,:,0:3]

    # add ten pixel to each dimension, to enhance the boundary condition
    shift = 10
    tmp_s_im_data = np.zeros( (s_im_data.shape[0]+2*shift, s_im_data.shape[1]+2*shift,3))
    tmp_s_im_data[shift:shift+s_im_data.shape[0], shift:shift+s_im_data.shape[1], :] = s_im_data
    s_im_data = tmp_s_im_data

    # basic information of the templates
    sh,sw,_ = s_im_data.shape
    mu, sigma, mask = get_mvn(s_im_data)

    b_im_data = skimage.io.imread(b_im_path)
    b_im_data = b_im_data[:,:,0:3]

    prob_mask = gen_prob( (mu,sigma, mask), b_im_data)
    log_prob_mask = np.log(prob_mask)
    max_pdf = multivariate_normal.pdf( mu, mean=mu, cov=sigma)
    log_minus1_prob_mask = np.log(max_pdf-prob_mask)
    #if True:
    if False:
        print max_pdf
        print log_prob_mask
        print np.sum(np.isnan(log_prob_mask))
        print log_minus1_prob_mask
        print np.sum(np.isnan(log_minus1_prob_mask))

        plt.subplot(121)
        plt.imshow(log_prob_mask)
        plt.subplot(122)
        plt.imshow(log_minus1_prob_mask)
        plt.show()
        return

    # iterate through all possible
    cache_path = './cache/%s_%d.npy'%(s_im_path,shift)
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
            print 'working on ',i
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
    #plt.imshow(score)
    #plt.show()

if __name__ == "__main__":
    #do_match('aimak5-1.png','afghanistan_ethnoling_97.png')
    #do_match('aimak1-1.png','afghanistan_ethnoling_97.png')
    #do_match('aimak2-1.png','afghanistan_ethnoling_97.png')
    #do_match('aimak3-1.png','afghanistan_ethnoling_97.png')
    #do_match('aimak4-1.png','afghanistan_ethnoling_97.png')
    #bid = 3
    #do_match('baloch%d-1.png'%(bid),'afghanistan_ethnoling_97.png')
    #do_match('sparse1-1.png','afghanistan_ethnoling_97.png')
    flist = os.listdir('./')
    for fname in flist:
        if fname.count('-1') > 0 and \
                fname.count('png') > 0 and \
                not os.path.isfile('./cache/%s_10.npy'%(fname))
            do_match(fname,'afghanistan_ethnoling_97.png')
