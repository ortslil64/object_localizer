# 3d object localizer #
By: Or Tslil
YOLO implementation: Zihao Zhang


This package fuses depth camera with RGB image to detect and localize objects in 3d. The detection is by yoloV3 implemented in tensorflow2. 


## Dependencies
The following python packges are required:
* numpy
* sklearn
* sciPy
* openCV
* TensorFlow 2.* (GPU version)
* currently tested in ros melodic in ubuntu 18.04

## Setup
1. Download repository to your catkin workspace:
```bash
git clone https://github.com/ortslil64/object_localizer.git
```
2. Build:
```bash
catkin_make
```
3. Install SSD image detector for ROS:
```bash
pip3 install -e object_localizer
```
4. Download pretrained weights from https://drive.google.com/file/d/1-Y-clzCjOboESm6be2RTAmVwDVo_D1kH/view?usp=sharing
5. Unzip the weights to `object_localizer/object_localizer/checkpoints/`
