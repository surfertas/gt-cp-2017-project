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



class ScaleExpansionDetector(object):
    def __init__(self, test=False):
        self.test = test

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

    def confirm_scale(self):
        pair2int = lambda pt: (int(pt[0]), int(pt[1]))
        rimg, cimg, _ = self.img_prv.shape
        for m in self.matches:
            r, c = pair2int(self.kp2[m.trainIdx].pt)
            size = int((self.kp2[m.trainIdx].size * 1.2 / 9 * 20) * 0.5)
            r0 = np.max([0, r - size])
            r1 = np.min([rimg, r + size])
            c0 = np.max([0, c - size])
            c1 = np.min([cimg, c + size])
            template1 = self.img_prv[r0:r1, c0:c1]
            for scale in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
                scaled = imresize(template1, scale)
                print scaled.shape

                # NOTE: incomplete, still WIP.

    def calculate_eps(self):
        pass
