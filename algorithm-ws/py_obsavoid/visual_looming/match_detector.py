import numpy as np
import scipy as sp
import cv2


class OrbTracker(object):
    def __init__(self):
        self.matches = None  # type: list of cv2.DMath
        self.kp1 = None  # type: list of cv2.KeyPoint items
        self.des1 = None  # type: numpy.ndarray of numpy.uint8 values.
        self.kp2 = None  # type: list of cv2.KeyPoint items.
        self.des2 = None  # type: numpy.ndarray of numpy.uint8 values.

        self.orb = cv2.orb = cv2.ORB_create(nfeatures=400, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
                                            patchSize=31, nlevels=8)  # alternate call required on some OpenCV versions
        # TODO See if higher nlevels value does some good.
        # TODO see if higher WTA_K does some good.
        # http: // docs.opencv.org / trunk / db / d95 / classcv_1_1ORB.html  # details

        self.bruteforce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # TODO see if knnmatch, FLANN could do some good
        # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#brute-force-matching-with-sift-descriptors-and-ratio-test

    def findMatchesBetweenImages(self, image_1, image_2):
        """ Return the list of matches between two input images.

        This function detects and computes ORB features from the
        input images, and returns the best matches using the normalized Hamming
        Distance.

        Parameters
        ----------
        image_1 : numpy.ndarray
            The first image (grayscale).

        image_2 : numpy.ndarray
            The second image. (grayscale).

        Returns
        -------
        image_1_kp : list
            The image_1 keypoints, the elements are of type cv2.KeyPoint.

        image_2_kp : list
            The image_2 keypoints, the elements are of type cv2.KeyPoint.

        matches : list
            A list of matches, length 10. Each item in the list is of type
            cv2.DMatch.
        """

        self.kp1, self.des1 = self.orb.detectAndCompute(image_1, None)
        self.kp2, self.des2 = self.orb.detectAndCompute(image_2, None)

        self.matches = self.bruteforce.match(self.des1, self.des2)
        self.matches = sorted(self.matches, key=lambda x: x.distance)

    def discard_miss_match(self, threshold):
        """ Filters the matches by distance attribute of the matches.
        Params:
            threshold - float: Threshold for match.distance.
        """
        self.matches = [m for m in self.matches if m.distance > threshold]

    def discard_size_thresh(self):
        """ Filters the matches by the size of the keypoints.
        """
        # queryIdx is 1st parameter, trainIdx is 2nd parameter
        self.matches = [m
                        for m in self.matches
                        if self.kp1[m.queryIdx].size > self.kp2[m.trainIdx].size
                        ]

