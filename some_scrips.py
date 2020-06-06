import cv2  
import glob
import os 
import numpy as  np 
UBIT='mgosi'
np.random.seed(sum([ord(c) for c  in  UBIT]))


def get_all_images(folder_path):
    """
    :param folder_path : path  to folder include images  for skitch 
    :return: list -> constant path  for all  image
    """
    list_path_images  =  glob.glob(os.path.join(folder_path,'*.*'))
    return list_path_images

def get_image(path_to_image):
    """

    :param path_to_image : path for image need read
    :return: image : type(numpy array)
    """
    return cv2.imread(path_to_image)

def get_sift_point(image):
    """
    :param path_to_image : path for image need read
    :return: image : type(numpy array)
    """
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (key_point, descs) = sift.detectAndCompute(gray, None)
    img =  cv2.drawKeypoints(gray, key_point,image)

    return key_point,descs , img


def BFmatch(desc1, desc2, kp_1,kp_2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(np.asarray(desc1,np.float32),np.asarray(desc2,np.float32),k=2)
    print("matches:",matches)
    #store  all  good match  as  per  lowe 's ratio test
    good_matches  =  []
    for  m,n  in  matches:
        if  m.distance < 0.7* n.distance:
            good_matches.append(m)
    print('good matches:',good_matches)
    src_pts = np.float32([ kp_1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)             #Gives us the index of the descriptor in the list of train descriptors 
    dst_pts = np.float32([ kp_2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)               #Gives us the index of the descriptor in the list of query descriptors 
    print("dst:",dst_pts)
    print("dst:",src_pts)
    #homography relates the transformation between two plane by using RANSAC Algorithm
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    return good_matches, matchesMask,H

