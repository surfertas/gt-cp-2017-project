import numpy as np
import scipy as sp
import cv2


class OrbTracker(object):
    def __init__(self):
        pass

    def findMatchesBetweenImages(self, image_1, image_2):
        """ Return the top 10 list of matches between two input images.

        This function detects and computes ORB features from the
        input images, and returns the best matches using the normalized Hamming
        Distance.

        Follow these steps:
        1. Compute ORB keypoints and descriptors for both images
        2. Create a Brute Force Matcher, using the hamming distance (and set
           crossCheck to true).
        3. Compute the matches between both images.
        4. Sort the matches based on distance so you get the best matches.
        5. Return the image_1 keypoints, image_2 keypoints, and the top 10 matches
           in a list.

        Note: We encourage you use OpenCV functionality (also shown in lecture) to
        complete this function.

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
        matches = None  # type: list of cv2.DMath
        image_1_kp = None  # type: list of cv2.KeyPoint items
        image_1_desc = None  # type: numpy.ndarray of numpy.uint8 values.
        image_2_kp = None  # type: list of cv2.KeyPoint items.
        image_2_desc = None  # type: numpy.ndarray of numpy.uint8 values.

        # orb = cv2.ORB()
        orb = cv2.orb = cv2.ORB_create(nfeatures=400, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
                                       patchSize=31, nlevels=8)  # alternate call required on some OpenCV versions


        # WRITE YOUR CODE HERE.
        image_1_kp, image_1_desc = orb.detectAndCompute(image_1, None)
        image_2_kp, image_2_desc = orb.detectAndCompute(image_2, None)

        # print "keypoints1:    ", len(image_1_kp), "    keypoints2:    ", len(image_2_kp)
        bruteforce = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bruteforce.match(image_1_desc, image_2_desc)

        matches = sorted(matches, key=lambda x: x.distance)

        # We coded the return statement for you. You are free to modify it -- just
        # make sure the tests pass.
        return image_1_kp, image_2_kp, matches[:10]

