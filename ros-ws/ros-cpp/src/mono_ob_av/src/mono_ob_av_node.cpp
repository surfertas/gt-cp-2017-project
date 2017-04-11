// Date:    4/11/2017

#include <ros/ros.h>
#include "mono_ob_av/MonoObsAvoid.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "mono_ob_av");
  ros::NodeHandle nodeHandle("~");

  mono_ob_av::MonoObsAvoid mono_cam_quad(nodeHandle, false);

  ros::spin();
  return 0;
}
