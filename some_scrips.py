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

def knn_match_images (good_match, matchesMask):
    ran_pts1 = []
    ran_pts2 = []
    for i in [np.random.randint(0,len(good_match) -1 ) for x in range(10)] :
        ran_pts1.append(good_match[i])
        ran_pts2.append(matchesMask[i])
    return ran_pts1 , ran_pts2

def warping(path_img_1 , path_img_2):

    img1  =  cv2.imread(path_img_1)
    img2  =  cv2.imread(path_img_2)

    key_point1,descs1 , img_1 =  get_sift_point(img1)
    key_point2,descs2 , img_2 =  get_sift_point(img2)
    print(key_point2)
    good_match, matchesMask,H = BFmatch( descs1,descs2,key_point1, key_point2)
    print (good_match)

    ran_pts1 , ran_pts2 = knn_match_images (good_match, matchesMask)

    height,width = img_1.shape[:2]
    pts = np.float32([ [0,0],[0,height-1],[width-1,height-1],[width-1,0] ]).reshape(-1,1,2)


    #Taking the border points of the image(i.e Corner Points)
    dst = cv2.perspectiveTransform(pts,H)

    #Extracting the rows and cols
    rows1, cols1 = img_1.shape[:2]
    rows2, cols2 = img_2.shape[:2]

    #Getting Border Points
    pts1 = np.float32([[0,0], [0,rows1], [cols1, rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

    #Taking Perspective
    pts2 = cv2.perspectiveTransform(temp_points, H)
    pts = np.concatenate((pts1, pts2), axis=0)
    #For calculating the size of the output image
    [x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)

    trans_dist = [-x_min, -y_min]

    H_trans = np.array([[1, 0, trans_dist[0]], [0, 1, trans_dist[1]], [0,0,1]])
    #Warping image 1 on top of Image 2
    warp_img = cv2.warpPerspective(img_1, H_trans.dot(H), (x_max - x_min, y_max - y_min))
    warp_img[trans_dist[1]:rows1+trans_dist[1],trans_dist[0]:cols1+trans_dist[0]] = img_2
    cv2.imwrite("final.jpg",warp_img)
        
if __name__ == "__main__":
    warping('/home/sink-all/Desktop/work/Collage_Panorama/data/mountain1.jpg','/home/sink-all/Desktop/work/Collage_Panorama/data/mountain2.jpg')
