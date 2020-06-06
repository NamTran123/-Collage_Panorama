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

