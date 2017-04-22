#!/usr/bin/env python
"""
The obstacle detector uses the scale expansion detector algorithm.
This algorithm matches, filters, and calculates, the expansion of relevant
ORB features in consecutive images.
"""
import os
import argparse

import errno
import cv2
import numpy as np
import scipy as sp
from scipy.misc import imresize


class ObstacleDetector(object):

    """ Detects on-coming obstacles applicable to monocular cameras. """

    def __init__(self, curimg, prvimg, matches, kp1, kp2, test=False):
        self.test = test
        self.img_cur = curimg
        self.img_prv = prvimg
        self.matches = matches
        self.kp1 = kp1
        self.kp2 = kp2
        self.pair2int = lambda pt: (int(pt[0]), int(pt[1]))
        self.obstacle_scale = []

    def _get_template_coord(self, rbnd, cbnd, keypoint, scale=1):
        """ Returns the corners of the template, relative to original image.
        Params:
            rbnd: Max row value of original image.
            cbnd: Max col value of original image.
            keypoint: keypoint used to get point, and size.
            scale: scale multiplier used to scale template.
        Returns:
            r0, r1, c0, c1: Corner values of template.
        """
        # Helper function
        c, r = self.pair2int(keypoint.pt)
        size = int((keypoint.size * 1.2 / 9 * 20) * scale // 2)
        r0 = np.max([0, r - size])
        r1 = np.min([rbnd, r + size])
        c0 = np.max([0, c - size])
        c1 = np.min([cbnd, c + size])
        return (r0, r1, c0, c1)

    def _filter_roi(self, rbnd, cbnd, keypoint):
        """ Tests if keypoint is in region of interest.
        Note: Uses 1.5 (max scale) for checking.

        Params:
            rbnd: Max row value of original image.
            cbnd: Max col value of original image.
            keypoint: keypoint used to get point, and size.
        Returns:
            True if keypoint is in ROI, else False.

        Note: Uses 1.5 (max scale) for checking.
        """
        # The assumption that we are making is that if keypoint is near edge
        # then not an obstacle of interest.
        c, r = self.pair2int(keypoint.pt)
        size = int((keypoint.size * 1.2 / 9 * 20) * 1.5 // 2)
        if (r - size < 0
            or r + size > rbnd
            or c - size < 0
                or c + size > cbnd):
            return False
        return True

    def confirm_scale(self):
        """ Scale algorithm that filters for scale variation.

        Creates a template from the previous image, and scales up the current
        image using different scale factors. The algorithm fitlers out matches
        that are determined not to be obstacles.
        """
        # Note: The paper uses a different function to compare the two
        # templates, where this implementation uses MSE.
        rimg, cimg, _ = self.img_prv.shape
        obstacle = []
        obstacle_scale = []
        for m in self.matches:

            # Skips if key point not in region of interest.
            if not self._filter_roi(rimg, cimg, self.kp2[m.trainIdx]):
                continue

            (
                t1r0, t1r1, t1c0, t1c1
            ) = self._get_template_coord(rimg, cimg, self.kp2[m.trainIdx])

            trntemplate1 = self.img_prv[t1r0:t1r1, t1c0:t1c1]
            TMmin = np.inf
            smin = 1.0
            # for scale in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            for scale in [1.0, 1.3, 1.5, 1.7, 1.9]:
                if not self._filter_roi(rimg, cimg, self.kp1[m.queryIdx]):
                    continue

                # trntemplate = imresize(trntemplate1, scale)
                trntemplate = cv2.resize(src=trntemplate1,
                                         dsize=(int(trntemplate1.shape[1] * scale), int(
                                             trntemplate1.shape[0] * scale)),
                                         interpolation=cv2.INTER_AREA)

                (
                    t2r0, t2r1, t2c0, t2c1
                ) = self._get_template_coord(rimg, cimg, self.kp1[m.queryIdx], scale)

                qrytemplate = self.img_cur[t2r0:t2r1, t2c0:t2c1]

                # qrytemplate = imresize(qrytemplate, (trntemplate.shape))
                qrytemplate = cv2.resize(src=qrytemplate,
                                         dsize=(
                                             trntemplate.shape[
                                                 0], trntemplate.shape[1]),
                                         interpolation=cv2.INTER_AREA)

                # RMS error between two images
                # TMscale = np.mean(np.square(trntemplate - qrytemplate))
                TMscale = ((trntemplate - qrytemplate) ** 2).mean(axis=None)

                if scale == 1.0:
                    TMone = TMscale

                if TMscale < TMmin:
                    TMmin = TMscale
                    smin = scale

            if smin > 1.2 and TMmin < 0.8 * TMone:
                # print smin
                obstacle.append(m)
                obstacle_scale.append(smin)
        self.matches = obstacle
        self.obstacle_scale = obstacle_scale

    def get_obstacle_position(self):
        """ Takes the average of the keypoint locations and returns the obstacle
        location as single point.

        Returns:    Obstacle point.
        """
        rtmp, ctmp = [], []
        for m in self.matches:
            c, r = self.pair2int(self.kp1[m.queryIdx].pt)
            rtmp.append(r)
            ctmp.append(c)
        return int(np.mean(ctmp)), int(np.mean(rtmp))
