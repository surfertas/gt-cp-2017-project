## Frontal obstacle avoidance using monocular camera for UAV

Implementation: Mori, T., Scherer, S. [First Results in Detecting and Avoiding Frontal Obstacles from
a Monocular Camera for Micro Unmanned Aerial
Vehicles](https://www-preview.ri.cmu.edu/pub_files/2013/5/monocularObstacleAvoidance.pdf)

### Introduction
When image of an object becomes increasingly large on the perceiver's retina, i.e., when object looms, we sense an approaching object. Visual looming is one of the monocular cues which can provide sense of depth. It can be combined with camera fps to find out TTC (time to contact) of the obstacle. However, calculation of TTC is, strictly speaking, perception of velocity rather than depth.

The implementation by by Mori, T. Scherer uses SURF to extract keypoints. Our implementation uses ORB (Oriented FAST and Rotated BRIEF) to extract keypoints and descriptors from image. 

#### Credits: 
Tasuku Miura, Dhiraj Dhule

### Requirements.
Python 2.7 and OpenCV3.1 (cv2 python) library are required.

### Usage
```bash
git clone https://github.com/surfertas/gt-cp-2017-project.git
cd gt-cp-2017-project/visual_looming
python main.py --debug True --skip 1
```
