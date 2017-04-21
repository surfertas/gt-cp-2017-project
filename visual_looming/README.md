To test,
edit main.py for camera settings.
Then run with debug set to true for test environment:
python main.py --debug True


Note:
Visualization: Currently visualizes kp matches, and obstacle location
(average of kp.pts). The previous image, has a history of the obstacle location
for easier visualization.

Todo:
1. Prepare separate test environment (on top of TrackingTest class) for direct camera, video file testing.
2. Integrate real time input to update the template so that we can do some basic testing first.
3. Simple video test case has been implemented but the overall result is very
slow. Is the visualization slowing down the program, or is it the original
algorithm. Need to do some testing here.
4. Need to implement algorithm 3 now, the controller in response to a detected
obstacle. The question is should the controller be responding to one obstacle
recognition, or a wait until confirmation of obstacle?

 
