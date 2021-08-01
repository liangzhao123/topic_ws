
import rospy

from sensor_msgs.msg import Image
from theora_image_transport.msg import Packet

import cv2
import ros_numpy
import torch
import os



class image_listenner:
    def __init__(self,image_save_dir):
        # self.image_sub = rospy.Subscriber("/pylon_camera_node/image_raw/rgb",Image,self.image_sub_callback)
        # self.image_sub = rospy.Subscriber("/pylon_camera_node/image_raw/theora",Image,self.image_sub_callback)
        self.imgage_sub_thera = rospy.Subscriber("/pylon_camera_node/image_raw/theora",Packet,self.image_sub_callback)
        self.save_dir = image_save_dir
    def image_sub_callback(self, data):
        ''' callback of image_sub '''
        filename = str(data.header.seq) + ".png"
        save_filename = os.path.join(self.save_dir,filename)
        # self.img = ros_numpy.numpify(data["data"])
        self.img = ros_numpy.numpify(data)
        # cv2.imwrite(save_filename,self.img)
        # print('filename')
        cv2.imshow("Image ", self.img)
        cv2.waitKey(10)

if __name__ == '__main__':
    rospy.init_node('image_listenner', anonymous=True)
    # image_save_dir = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/rosbag/leishen/2020-12-04-17-00-15-image"
    image_save_dir = "/media/liang/Elements/Five_Lidar_data/beike/image"
    image_listenning_ = image_listenner(image_save_dir=image_save_dir)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
