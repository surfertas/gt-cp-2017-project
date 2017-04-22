import argparse
import timeit

import numpy as np
import cv2

import match_detector as orb  # code from assignment 7
from visualizer import draw_output, draw_matches
from obstacle_detector import ObstacleDetector


class TrackingTest(object):

    """ Extends all pieces needed for experimentation.

    Test case for video, consecutive images can be implemented.
    """

    def __init__(self, cap_dev, tracker, visualation, dist_thresh, debug=False):
        self.cap = cv2.VideoCapture(cap_dev)
        self.orb = tracker()
        self.template = None
        self.visual = visualation
        self.debug = debug
        self.dist_thresh = dist_thresh
        self.setup_camera()
        self.template_cap_time = None

    def setup_camera(self, width=1280, height=720, fps=10):
        """ To configure for specific cam.

        Params:
            width: Width dimension of video capture.
            height: Height dimension of video capture.
            fps: Frame per seconds attribute setting for video capture.
        Note: Defaults set for Intel Realsense camera
        """
        # Slow down fps for debugging.
        fps = 1 if self.debug else fps

        self.cap.set(3, width)   # width
        self.cap.set(4, height)  # height
        self.cap.set(5, fps)     # fps
        self.cap.set(16, 1)      # convert RGB

    def update_template(self):
        """ Updates the template captured from video.

        Note: this function would be useful for real time update
        of the template and multi instance version of application.
        """
        for i in range(20):
            ret, temp = self.cap.read()
        if ret:
            self.template = temp
            print "template captured"
            return self.template
        else:
            print "template update failed"
            return None

    def process_next_image(self, img):
        """ Process input image and template to get matches using tracking algorithm.

        Params:
            img: New image frame.
        Returns:
            Matched image with the template

        Note: image 1 = current image, image 2 = previous image.
        """
        self.orb.find_matches_between(img, self.template)

        self.orb.discard_miss_match(threshold=self.dist_thresh)
        self.orb.discard_size_thresh()

        detector = ObstacleDetector(
            img, self.template, self.orb.matches, self.orb.kp1, self.orb.kp2)
        detector.confirm_scale()

        if detector.matches:
            # find speed of approaching object.
            # assuming that object becomes 1.5 times larger when comes from 3
            # meters to 2 meters.
            time_since_template_captured = timeit.default_timer(
            ) - self.template_cap_time
            avg_scale = np.mean(np.array(detector.obstacle_scale, dtype=float))
            dist_traveled = avg_scale / 1.5
            speed = dist_traveled / time_since_template_captured
            speed = float("{0:.2f}".format(speed)) * 100

            output = img[:, :]

            # add speed overlay
            font = cv2.FONT_HERSHEY_SIMPLEX

            output = cv2.putText(
                output, 'Obstacle_speed: ' +
                    str(speed) + " cm/s ", (10, 30), font, 1, (200, 255, 155),
                                 2,
                                 cv2.LINE_AA)

            obstacle = detector.get_obstacle_position()
            cv2.circle(img, obstacle, 5, (0, 255, 0), thickness=5)
            cv2.circle(self.template, obstacle, 5, (0, 255, 0), thickness=5)

            draw_output(
                self.orb.matches, detector.matches, self.orb.kp1, output)

        annotated_matches = None
        annotated_matches = self.visual(
            img, self.orb.kp1, self.template, self.orb.kp2, detector.matches,
            annotated_matches, flags=2)

        return annotated_matches

    def skip_frames(self, frames):
        """ Set to skip few frames from the webcam.

        Params:
            frames: Number of frames to skip.
        """
        for i in range(frames):
            ret, template = self.cap.read()

    def grab_next_img(self):
        """ Function to grab next image from self.cap.

        Returns:
            frame: Returns capture frame, else None.
        """
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None


# Testing environments:
#   Available environments are camera at device 0, and video (mp4, etc).
def test_on_camera(dist_thresh, fps, debug=False):
    """ Use this setup method to setup everything on the live camera input.

    Params:
        dist_thresh: Threshold passed to orb tracker to filter matches.
        fps: Frames per second passed to video capture.
        debug: Set to True if in debug environment.
    """
    test = TrackingTest(
        0, orb.OrbTracker, cv2.drawMatches, dist_thresh=dist_thresh, debug=True)
    test.skip_frames(10)
    test.update_template()
    test.template_cap_time = timeit.default_timer()
    elapsed = 0
    while test.cap.isOpened():
        test.skip_frames(int(fps * elapsed))
        start_time = timeit.default_timer()

        img = test.grab_next_img()
        match = test.process_next_image(img)

        if debug:
            # resize image
            scale_factor = 0.5
            resized_match = cv2.resize(
                src=match, dsize=(
                    int(match.shape[1] * scale_factor), match.shape[0]),
                interpolation=cv2.INTER_AREA)
            cv2.imshow("matches", resized_match)
        elapsed = timeit.default_timer() - start_time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    test.cap.release()
    cv2.destroyAllWindows()


def test_on_video(video, dist_thresh, skip, debug=False):
    """ Setup method to setup everything on the video file input

     Params:
        video: Video file to read.
        dist_thresh: Threshold passed to orb tracker to filter matches.
        skip: # of frames to skip for processing.
        debug: Set to True if in debug environment.
    """
    test = TrackingTest('./test_vids/{}'.format(video), orb.OrbTracker,
                        cv2.drawMatches, dist_thresh=dist_thresh, debug=True)
    test.skip_frames(10)
    test.update_template()

    # avg_fps = 0
    # fps_records = []
    test.template_cap_time = timeit.default_timer()

    while test.cap.isOpened():
        test.skip_frames(skip)
        # start_time = timeit.default_timer()

        img = test.grab_next_img()
        match = test.process_next_image(img)
        cv2.imshow("matches", match)

        # elapsed = timeit.default_timer() - start_time
        # fps_records.append(elapsed)
        # avg_fps = 1.0 / (np.mean(np.array(fps_records, dtype=np.float)))
        # print "time for loop ", elapsed, "avg fps ", avg_fps

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    test.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Monocular Obstacle Avoidance')
    parser.add_argument('--thresh', '-t', type=float, default=0.25,
                        help='Sets the distance threshold for match filtering')
    parser.add_argument('--debug', '-d', type=str, default=False,
                        help='Sets real time camera.')
    parser.add_argument('--skip', '-s', type=int, default=0,
                        help='Sets number of frames to skip for processing')
    parser.add_argument('--video', '-v', type=str, default="1.mp4",
                        help='Specifies the video to use for testing.')

    args = parser.parse_args()

    # for camera testing pass actual fps of camera instead of skip
    # test_on_camera(args.thresh, args.skip, args.debug)

    # for video testing the skip arg will decide obstacle speed, because total time since template capture is used
    # to calculate the speed of obstacle.
    test_on_video(args.video, args.thresh, args.skip, args.debug)
