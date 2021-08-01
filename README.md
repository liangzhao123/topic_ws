# topic_ws| [blog](https://blog.csdn.net/liang_shuaige/article/details/114854061?spm=1001.2014.3001.5501)
ROS implementation of IOU-SSD, deep learning based 3D object detection algorithm 
<<<<<<< HEAD


# 3D object detection tools
 author: [liangzhao](https://github.com/liangzhao123)
## Using ROS achieve online detection    
This is the official implementation of Seg-RCNN and IOU-SSD through  ROS

### introduction 
Deep learning based 3D obejct detection.


src/object_detect
### installation 
(1) Clone this repository.
```
git  clone <>
```
(2) Setup Python environment.
```
virtualenv -p /usr/bin/python3.6   venv
source venv/bin/activate 
pip install -r src/requirement.txt 
```
(3) install cuda cudnn nvidia dirver .etc
```
pip install torch==1.4.0 torchvison=0.5.0 etc 
```
(4) 

install spconv 
install ros_numpy please search in github
install c++ function in topic_ws/src/bject_detect/for_ros/iou3d_nm
install c++ function in topic_ws/src/object_detect/for_ros/pointnet2

# usage
train on KITTI/Leishen dataset.

test on KITTI/Leishen rosbag
## prepare data
### (1) download rosbag.(1)kitti rosbag(2) leishen rosbag

### (2) download kitti dataset
[baidudisk](https://pan.baidu.com/s/1lDRciFN2HLREZVaE6E2Brg)   
password:`w0fg`
## online detection
(1) run model in leishen rosbag
firts play a leishen rosbag , then 
```
cd topic_ws
source venv/bin/activate
source catkin_workspace/install/setup.bash
cd src/object_detect/leishen/single_stage_model
python IOU_SSD_leishen_detection.py
```
or
```
cd topic_ws
source venv/bin/activate
source catkin_workspace/install/setup.bash
cd src/object_detect/tools/
python IOU_SSD_detection.py 
```

(2)run model in kitti rosbag
firts play a leishen rosbag , then
```
cd topic_ws
source venv/bin/activate
cd src/object_detect/for_ros
python real_time_detect.py
```


=======
## 作者最近在改硕士大论文，代码将在未来几个周整理好发布
# The code will release soon.
### [blog](https://blog.csdn.net/liang_shuaige/article/details/114854061?spm=1001.2014.3001.5501)
>>>>>>> e804cc467b17f602814cbad31247dcd07e335bfd
