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

        ## LEAVE THIS ORB STATEMENT COMMENTED OUT UNTIL AFTER YOU ARE DONE WITH AUTOGRADING.
        ## This is for the Investigation part of the assignment only.
        # orb = cv2.ORB(nfeatures=500, scaleFactor=1.2, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE,
        #               patchSize=31)

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

    def drawMatches(self, image_1, image_1_keypoints, image_2, image_2_keypoints, matches):
        """ Draws the matches between the image_1 and image_2.

        Note: Do not edit this function, it is provided for you for visualization
        purposes.

        Args:
        image_1 (numpy.ndarray): The first image (can be color or grayscale).
        image_1_keypoints (list): The image_1 keypoints, the elements are of type
                                  cv2.KeyPoint.
        image_2 (numpy.ndarray): The image to search in (can be color or grayscale)
        image_2_keypoints (list): The image_2 keypoints, the elements are of type
                                  cv2.KeyPoint.

        Returns:
        output (numpy.ndarray): An output image that draws lines from the input
                                image to the output image based on where the
                                matching features are.
        """
        # Compute number of channels.
        num_channels = 1
        if len(image_1.shape) == 3:
            num_channels = image_1.shape[2]
        # Separation between images.
        margin = 10
        # Create an array that will fit both images (with a margin of 10 to
        # separate the two images)
        joined_image = np.zeros((max(image_1.shape[0], image_2.shape[0]),
                                 image_1.shape[1] + image_2.shape[1] + margin,
                                 3))
        if num_channels == 1:
            for channel_idx in range(3):
                joined_image[:image_1.shape[0],
                :image_1.shape[1],
                channel_idx] = image_1
                joined_image[:image_2.shape[0],
                image_1.shape[1] + margin:,
                channel_idx] = image_2
        else:
            joined_image[:image_1.shape[0], :image_1.shape[1]] = image_1
            joined_image[:image_2.shape[0], image_1.shape[1] + margin:] = image_2

        for match in matches:
            image_1_point = (int(image_1_keypoints[match.queryIdx].pt[0]),
                             int(image_1_keypoints[match.queryIdx].pt[1]))
            image_2_point = (int(image_2_keypoints[match.trainIdx].pt[0] +
                                 image_1.shape[1] + margin),
                             int(image_2_keypoints[match.trainIdx].pt[1]))

            rgb = (np.random.rand(3) * 255).astype(np.int)
            cv2.circle(joined_image, image_1_point, 5, rgb, thickness=-1)
            cv2.circle(joined_image, image_2_point, 5, rgb, thickness=-1)
            cv2.line(joined_image, image_1_point, image_2_point, rgb, thickness=3)

        return joined_image
