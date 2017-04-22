from main import *
from unittest import TestCase, main

class TestVisualLooming(TestCase):
    
    def test_video(self):
        """ Tests if system works on video input """
        return test_on_video("./input_video/1.mp4", 0.25, 0, True)

