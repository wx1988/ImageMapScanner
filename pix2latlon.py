#from sklearn.svm import SVR
#from sklearn.linear_model import LogisticRegression
import numpy as np
import copy

ref_list = [ [[353,617], [64,32]], 
        [[615,620],[68,32]],
        [[615,310],[68,36]],
        [[361,307],[64,36]] ]

ref_list2 = [ [[66,293],[np.nan, 36]], 
        [[67,604],[np.nan, 32]], 
        [[1117,294],[np.nan, 36]],
        [[1117,605],[np.nan, 32]],
        [[122,60],[60, np.nan]], 
        [[368,61],[64, np.nan]], 
        [[613,61],[68, np.nan]], 
        [[859,61],[72, np.nan]],
        [[76,868],[60, np.nan]],
        [[346,869],[64, np.nan]],
        [[615,869],[68, np.nan]],
        [[884,869],[72, np.nan]]
        ]

test_list = [ [[109,296], [60,36]], 
[[866,306], [72,36]],
[[91,606], [60,32]],
[[876,616], [72,32]]]

def get_train_data():
    tmp_ref_list = copy.copy(ref_list)
    tmp_ref_list.extend(ref_list2)
    fv_list = np.array( [pix for pix, ll in tmp_ref_list] )
    long_list = np.array( [ll[0] for pix, ll in tmp_ref_list] )
    lat_list = np.array( [ll[1] for pix, ll in tmp_ref_list] )
    long_pos = np.logical_not(np.isnan(long_list))
    lat_pos = np.logical_not(np.isnan(lat_list))
    #print long_pos, lat_pos, np.where(long_pos)[0]
    long_fv_list = fv_list[ np.where(long_pos)[0], :]  
    lat_fv_list = fv_list[ np.where(lat_pos)[0], :]  
    long_list = long_list[ np.where(long_pos)[0] ]
    lat_list = lat_list[ np.where(lat_pos)[0] ]
    return long_fv_list, long_list, lat_fv_list, lat_list

def ml_pix2ll(pts_list):
    print 'pred'
    # TODO, might just use a non-linear regressor for this
    long_svr = SVR(kernel='poly')
    lat_svr = SVR(kernel='poly')

    #long_svr = LogisticRegression()
    #lat_svr = LogisticRegression()
    long_fv_list, long_list, lat_fv_list, lat_list =\
            get_train_data()

    p_long_list = long_svr.fit(long_fv_list, long_list).predict(pts_list)
    p_lat_list = lat_svr.fit(lat_fv_list, lat_list).predict(pts_list)
    print p_long_list
    print p_lat_list

def pix2ll(pts_list):
    # NOTE, only work with one vertical line
    #4 degreee
    # based on 32 degree data, check where the 36 degree data
    #[[353,617], [64,32]], 
    #[[615,620],[68,32]],
    p1 = np.array(ref_list[0][0])
    p2 = np.array(ref_list[1][0])
    midp = [np.mean([p1[0],p2[0]]), np.mean([p1[1],p2[1]])]
    #print midp
    length = np.sqrt(np.dot(p1-p2, p1-p2))
    
    r = length/2/ np.sin(2./180*np.pi) 
    d = r* 4./(90-32)
    #print r,d

    #get the original
    org = [ p2[0], p2[1]-r]

    #calculate the distance, 
    # TODO, more mathematicall way
    r_lat_list = []
    r_long_list = []
    for i in range(len(pts_list)):
        tmp_v = org - pts_list[i,:]
        tmp_d = np.sqrt( np.dot(tmp_v,tmp_v) )
        lat = 90 - tmp_d / (r / (90.-32))

        hd = pts_list[i,0] - org[0]
        vd = pts_list[i,1] - org[1]
        lon = 68 + np.arctan( hd/vd ) / np.pi * 180
        
        r_lat_list.append(lat)
        r_long_list.append(lon)
    #print r_lat_list
    #print r_long_list
    res = np.zeros_like( pts_list).astype(np.float32)
    #print res.shape, pts_list.shape
    res[:,0] = np.array(r_long_list)
    res[:,1] = r_lat_list
    #pass
    #print res
    return res

if __name__ == "__main__":
    #get_train_data()
    #exit()
    pix_list = [pix for pix,ll in test_list]
    ll_list = [ll for pix,ll in test_list]
    pix2ll(np.array(pix_list))
    
