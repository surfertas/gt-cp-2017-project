#!/usr/bin/env python
"""
The scale expansion detector algorithm. This algorithm matches,
filters, and calculates, the expansion of relevant ORB features in consecutive
images.
"""
import os
import argparse

import errno
import cv2
import numpy as np
import scipy as sp
from scipy.misc import imresize

from algo_util import show_kp

# import template matching algorithm

SRC_FOLDER = "input/test_diff"
OUT_FOLDER = "output"
EXTENSIONS = set(["jpeg", "jpg", "png"])


class ScaleExpansionDetector(object):

    def __init__(self, test=False):
        self.test = test        # true if in testing env
        self.img_cur = None
        self.img_prv = None
        self.matches = None     # type: list of cv2.DMath
        self.kp1 = None         # type: list of cv2.KeyPoint items
        self.des1 = None        # type: numpy.ndarray of numpy.uint8 values.
        self.kp2 = None         # type: list of cv2.KeyPoint items.
        self.des2 = None        # type: numpy.ndarray of numpy.uint8 values.
        self.pair2int = lambda pt: (int(pt[0]), int(pt[1]))

        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.orb = cv2.ORB_create(nfeatures=500,
                                  scaleFactor=1.2,
                                  WTA_K=2,
                                  scoreType=cv2.ORB_HARRIS_SCORE,
                                  patchSize=31)

    def get_orb_matches(self, img_cur, img_prv):
        """ Finds keypoints, matches between current image and previous image.

        Params:
            img_cur - numpy.darray: Current image (grayscale)
            img_prv - numpy.darray: Previous image (grayscale)
        """
        self.img_cur = img_cur
        self.img_prv = img_prv

        self.kp1, self.des1 = self.orb.detectAndCompute(self.img_cur, None)
        self.kp2, self.des2 = self.orb.detectAndCompute(self.img_prv, None)

        self.matches = sorted(self.bf.match(self.des1, self.des2),
                              key=lambda m: m.distance)
        if self.test:
            show_kp(
                self.img_cur, self.kp1, self.img_prv, self.kp2, self.matches,
                    "matches", OUT_FOLDER)

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

    def get_template_coord(self, rbnd, cbnd, keypoint, scale=1):
        # Helper function
        r, c = self.pair2int(keypoint.pt)
        size = int((keypoint.size * 1.2 / 9 * 20) * scale//2)
        r0 = np.max([0, r - size])
        r1 = np.min([rbnd, r + size])
        c0 = np.max([0, c - size])
        c1 = np.min([cbnd, c + size])
        return (r0, r1, c0, c1)

    def confirm_scale(self):
        rimg, cimg, _ = self.img_prv.shape
        obstacle = []
        for m in self.matches:
            (
            t1r0, t1r1, t1c0, t1c1
            ) = self.get_template_coord(rimg, cimg, self.kp2[m.trainIdx])
            
            trntemplate = self.img_prv[t1r0:t1r1, t1c0:t1c1]
            TMmin = np.inf
            smin = 1.0
            for scale in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
                trntemplate = imresize(trntemplate, scale)
                (
                t2r0, t2r1, t2c0, t2c1
                ) = self.get_template_coord(rimg, cimg, self.kp1[m.queryIdx], scale)

                qrytemplate = self.img_cur[t2r0:t2r1, t2c0:t2c1]
                qrytemplate = imresize(qrytemplate, (trntemplate.shape))

                TMscale = np.mean(np.square(trntemplate - qrytemplate))

                if scale == 1.0:
                    TMone = TMscale
                
                if TMscale < TMmin:
                    TMmin = TMscale
                    smin = scale
            
            # NOTE: Paper uses 0.8 * TMone but that results in zero obstacles.
            if smin > 1.2 and TMmin < 1.0*TMone:
                obstacle.append(m)
                
        self.matches = obstacle

    def get_obstacle_position(self):
        rtmp, ctmp = [],[]
        for m in self.matches:
            r, c = self.pair2int(self.kp1[m.queryIdx].pt)
            rtmp.append(r)
            ctmp.append(c)
        return int(np.mean(rtmp)), int(np.mean(ctmp))
        
        


# Helper functions for prototyping
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
    parser = argparse.ArgumentParser(
        description='Monocular Obstacle Avoidance')
    parser.add_argument('--dist_thresh', '-d', type=float, default=0.25,
                        help='Sets the distance threshold for match filtering')
    args = parser.parse_args()

    # Only two images in test folder to index [0] needed.
    flight_imgs = load_images()
    curr = flight_imgs[:-1]
    prev = flight_imgs[1:]

    curprev_tpl = zip(curr, prev)
    algo = ScaleExpansionDetector(test=True)

    # Just for testing, checking if kp are being filtered.
    for pair in curprev_tpl:
        algo.get_orb_matches(*pair)
        print "after orb: {}".format(len(algo.matches))
        algo.discard_miss_match(args.dist_thresh)
        print "after miss: {}".format(len(algo.matches))
        algo.discard_size_thresh()
        print "after_size: {}".format(len(algo.matches))
        algo.confirm_scale()
        print "after_size: {}".format(len(algo.matches))


if __name__ == "__main__":
    main()
