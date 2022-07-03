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

## virtual environment
```
conda create -n aerials tensorflow-gpu=1.14 cudatoolkit=10.0 python=3.7
```
```
conda activate aerials-env
```
```
conda install -c conda-forge keras=2.2.5
pip install keras-segmentation
pip install numpy
pip install scipy matplotlib pillow
pip install imutils h5py==2.10.0 requests progressbar2
pip install cython
pip install scikit-learn scikit-build scikit-image
pip install opencv-contrib-python==4.4.0.46
pip install tensorflow-gpu==1.14.0
pip install keras==2.2.5
pip install opencv-python==4.4.0.42
pip install keras-segmentation
pip install rospkg empy
pip install matplotlib
pip install 'gast==0.2.2'
pip install opencv-python-headless==4.1.2.30
pip install albumentations==0.1.12
```

## use jupyter notebook
```
pip install notebook
```
Run:
```
jupyter notebook

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