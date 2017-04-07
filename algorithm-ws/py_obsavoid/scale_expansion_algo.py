#!/usr/bin python
"""
Algorithm 1: The scale expansion detector algorithm. This algorithm matches,
filters, and calculates, the expansion of relevant ORB features in consecutive
images.
"""
import numpy as np
import scipy as sp
import cv2

#import template matching algorithm


class ScaleExpansionDetector(object):
    
    def __init__(self):
        #Need to figure out what to share, kp?
        #array of keypoints?

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
#1. load images



def main():
    pass


if __name__=="__main__":
    main()

