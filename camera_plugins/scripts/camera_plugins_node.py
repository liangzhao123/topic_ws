#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import ros_numpy

import cv2

class image_listenner:
    def __init__(self): 
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/pylon_camera_node/image_raw",Image,self.image_sub_callback)
        self.image_numpy_pub = rospy.Publisher('/pylon_camera_node/image_raw/rgb', Image, queue_size=1)

    def image_sub_callback(self, msg):
        ''' callback of image_sub '''
        try:
            self.img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            new_msg = ros_numpy.msgify(Image, self.img, encoding='bgr8')
            new_msg.header.seq = msg.header.seq
            new_msg.header.stamp = msg.header.stamp
            new_msg.header.frame_id = "image_rgb"
            self.image_numpy_pub.publish(new_msg)
            print(new_msg.header.seq)

        except CvBridgeError as e:
            print(e) 

if __name__ == '__main__':
    rospy.init_node('image_listenner', anonymous=True)
    image_listenning = image_listenner()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
