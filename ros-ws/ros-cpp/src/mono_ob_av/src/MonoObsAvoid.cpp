// 
// Date: 4/12/2017
#include "MonoObsAvoid.hpp"

namespace mono_ob_av {

    MonoObsAvoid::MonoObsAvoid(ros::NodeHandle& nh) :
        nh_(nh)
    {
        registerPublisher();
        registerSubscriber();
        cout << "Node launched." << endl;
    }
    
    MonoObsAvoid::~MonoObsAvoid()
    {
    }
    
    void MonoObsAvoid::registerPublisher()
    {
    }

    void MonoObsAvoid::registerSubcriber()
    {
    }

    MonoObsAvoid::getCameraImage()
    {
        //subscribe to camera image publisher, published by different topic.
        //preprocess and publish
    }
