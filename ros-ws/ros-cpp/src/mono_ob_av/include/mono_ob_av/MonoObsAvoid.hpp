#ifndef MONOOBSAVOID_HPP
#define MONOOBSAVOID_HPP

#include <ros/ros.h>

namespace mono_ob_av {
    
    class MonoObsAvoid {
    // Class containing Monocular Obstacle Avoidance related functions and
    // attributes.
    public:
        MonoObsAvoid(ros::NodeHandle&);
    
        virtual ~MonoObsAvoid();

    private:
        ros::NodeHandle nh_;
    }
}    
    

