import argparse

import numpy as np
import cv2

import match_detector as orb  # code from assignment 7
from visualizer import show_kp, drawMatches


class TrackingTest(object):
    """extends all pieces needed for experimentation so that any test case can be implemented"""

    def __init__(self, cap_dev, tracker, visualation, debug=False):
        self.cap = cv2.VideoCapture(cap_dev)
        self.setup_camera()
        # todo come down to standard specification of tracker class
        self.orb = tracker()
        self.template = None
        self.visual = visualation
        self.debug = debug

    def setup_camera(self, width=1920, height=1280, fps=30):
        """ to configure for specific cam. Defaults set for Intel Realsense camera """
        self.cap.set(3, width)  # width
        self.cap.set(4, height)  # height
        self.cap.set(5, fps)  # fps
        self.cap.set(16, 1)  # convert RGB

    def update_template(self):
        """this function would be useful for real time update
         of the template and multi instance version of application"""

        ret, temp = self.cap.read()
        if ret:
            self.template = temp
            return self.template
        else:
            print "template update failed"
            return None

    def process_next_image(self, img):
        """
        Process input image and template to get matches using tracking algorithm.
        :param new image frame
        :return Matched image with the template
        For now let's use the drawMatches from a7. Going forward we will need better method.
        """
        k1, k2, matches = self.orb.findMatchesBetweenImages(self.template, img)
        annotated_matches = self.visual(self.template, k1, img, k2, matches).astype(np.uint8)
        return annotated_matches

    def skip_frames(self, frames):
        """useful for skipping first few frames from the webcam"""
        for i in range(frames):
            ret, template = self.cap.read()

    def grab_next_img(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None


def test_on_camera(dist_thresh, debug=False):
    """ use this setup method to setup everything on the live camera input"""
    test = TrackingTest(0, orb.OrbTracker, show_kp)
    test.skip_frames(10)
    test.update_template()
    while test.cap.isOpened():
        img = test.grab_next_img()
        match = test.process_next_image(img)
        if debug:
            cv2.imshow("matches", match)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    test.cap.release()
    cv2.destroyAllWindows()


def test_on_video(dist_thresh, debug=False):
    """use this setup method to setup everything on the video file input"""
    test = TrackingTest('video_file.mp4', orb.OrbTracker, drawMatches)
    test.skip_frames(10)
    test.update_template()
    while test.cap.isOpened():
        img = test.grab_next_img()
        match = test.process_next_image(img)
        cv2.imshow("matches", match)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    test.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Monocular Obstacle Avoidance')
    parser.add_argument('--dist_thresh', '-d', type=float, default=0.25,
                        help='Sets the distance threshold for match filtering')
    args = parser.parse_args()

    test_on_camera(args.dist_thresh, debug=True)
