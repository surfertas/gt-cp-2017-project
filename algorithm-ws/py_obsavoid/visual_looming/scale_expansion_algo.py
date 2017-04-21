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
        self.pair2int = lambda pt: (int(pt[0]), int(pt[1]))


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
        r, c = self.pair2int(keypoint.pt)
        size = int((keypoint.size * 1.2 / 9 * 20) * scale // 2)
        r0 = np.max([0, r - size])
        r1 = np.min([rbnd, r + size])
        c0 = np.max([0, c - size])
        c1 = np.min([cbnd, c + size])
        return (r0, r1, c0, c1)

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
            if smin > 1.2 and TMmin < 1.0 * TMone:
                obstacle.append(m)

        self.matches = obstacle

    def get_obstacle_position(self):
        """ Takes the average of the keypoint locations and returns the obstacle
        location as single point.
        Returns:    Obstacle point.
        """
        rtmp, ctmp = [], []
        for m in self.matches:
            r, c = self.pair2int(self.kp1[m.queryIdx].pt)
            rtmp.append(r)
            ctmp.append(c)
        return int(np.mean(rtmp)), int(np.mean(ctmp))
