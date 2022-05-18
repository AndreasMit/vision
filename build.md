# setup workspace

## download workspace
```
cd ~
git clone https://github.com/AndreasMit/vision.git
```

## vision workspace compile for python3 scripts:
```
cd ~/vision
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

## dependencies
```
sudo apt install python3-catkin-pkg-modules python3-rospkg-modules python3-empy
```

## image detection packages: (for reference)
you need the package cv_bridge to convert sensor_msgs/Image of ROS to opevCV format: http://wiki.ros.org/vision_opencv
```
cd ~/catkin_ws/src
git clone https://github.com/OTL/cv_camera.git
git clone https://github.com/ros-perception/image_common.git
git clone https://github.com/ros-perception/vision_opencv.git # only this is needed for now
git clone https://github.com/amc-nu/RosImageFolderPublisher.git
```