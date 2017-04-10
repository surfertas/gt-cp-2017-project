#!/usr/bin python
"""
Algorithm 1: The scale expansion detector algorithm. This algorithm matches,
filters, and calculates, the expansion of relevant ORB features in consecutive
images.
"""
import os
import errno
import numpy as np
import scipy as sp
import cv2

#import template matching algorithm

SRC_FOLDER = "input/test"
OUT_FOLDER = "output"
EXTENSIONS = set(["jpeg", "jpg", "png"])


class ScaleExpansionDetector(object):
    
    def __init__(self):
        #Need to figure out what to share, kp?
        #array of keypoints?
        pass

    def get_orb_matches(self, img_cur, img_prv):
        """
        Finds keypoints and matches between current image and previous image.

        Params: 
            img_cur - numpy.darray: Current image (grayscale)
            img_prv - numpy.darray: Previous image (grayscale)
        Returns:
            kp1 - [cv2.Keypoint]: Keypoints for current image.
            kp2 - [cv2.Keypoint]: Keypoints for previous image.
            matches - [cv2.Dmath]: All keypoint matches found.
        """
        matches = None      # type: list of cv2.DMath
        kp1 = None          # type: list of cv2.KeyPoint items
        des1 = None         # type: numpy.ndarray of numpy.uint8 values.
        kp2 = None          # type: list of cv2.KeyPoint items.
        des2 = None         # type: numpy.ndarray of numpy.uint8 values.

        orb = cv2.ORB_create(nfeatures=500, 
                             scaleFactor=1.2, 
                             WTA_K=2, 
                             scoreType=cv2.ORB_HARRIS_SCORE, 
                             patchSize=31)

        kp1, des1 = orb.detectAndCompute(img_cur, None)
        kp2, des2 = orb.detectAndCompute(img_prv, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key = lambda m: m.distance)

        return kp1, kp2, matches

    def discard_miss_match(self):
        pass

    def discard_size_thresh(self):
        pass

    def confirm_scale(self):
        pass

    def calculate_eps(self):
        pass


#Helper functions for prototyping
def load_images(): 
    """
    Loads images from SRC_FOLDER and creates an array of images, with allowed
    extensions defined in EXTENSIONS.
    Returns:
        img_stack - [image]: List of images.
    """
    src_contents = os.walk(SRC_FOLDER)
    dirpath, _, fnames = src_contents.next()

    image_dir = os.path.split(dirpath)[-1]
    output_dir = os.path.join(OUT_FOLDER, image_dir)

    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    image_files = sorted([os.path.join(dirpath, name) for name in fnames])

    img_stack = [cv2.imread(name) for name in image_files
                if os.path.splitext(name)[-1][1:].lower() in EXTENSIONS]

    return img_stack

def main():
    #  Loads images correctly
    proxy_flight_imgs = load_images()
    print(len(proxy_flight_imgs)) 


if __name__=="__main__":
    main()

