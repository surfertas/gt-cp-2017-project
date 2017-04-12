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
    
        
        getCameraImage();
    private:
        void registerPublisher();
        void registerSubscriber();
        
        ros::NodeHandle nh_;
    }
}    
    

