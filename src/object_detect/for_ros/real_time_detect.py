import sys
from pathlib import Path
sys_path = str((Path(__file__).resolve().parent / '../').resolve())
# sys.path.append("/home/liang/topic_ws/src/object_detect")
sys.path.append(str(sys_path))
import torch
import spconv
import numpy as np
import os
import datetime
from ros_numpy import point_cloud2
from numpy.lib.recfunctions import structured_to_unstructured

import for_ros.utils.calibration as calibration
from for_ros.utils.model import DetNet
from for_ros.utils.common_function import boxes3d_to_corners3d_lidar_torch

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import time
from visualization_msgs.msg import *
from geometry_msgs.msg import *
from for_ros.utils.common_function import publish_pc
from for_ros.utils.common_utils import mask_points_by_range
from for_ros.utils import common_utils

class PrepocessData:
    def __init__(self):
        # self.voxel_generator_cfg = cfg.DATA_CONFIG.VOXEL_GENERATOR
        self.voxel_generator = spconv.utils.VoxelGeneratorV2(
        voxel_size=[0.05,0.05,0.1],
        point_cloud_range=[0,-40.0,-3.0,70.4,40.0,1.0],
        max_num_points=5,
        max_voxels=16000
    )
        self.image_shape = np.array([375,1242],dtype=np.int32)
        self.sparse_shape = self.voxel_generator.grid_size[::-1] + [1, 0, 0]
        self.calib_data = self.get_calib()

    def get_calib(self):
        # calib_file = os.path.join("/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/training/", 'calib', '000137.txt' )
        calib_file = "calib.txt"
        assert os.path.exists(calib_file)
        return calibration.Calibration(calib_file)

    def points2voxel(self,points):
        voxel_grid = self.voxel_generator.generate(points)
        input_dict = {}
        input_dict["voxels"] = voxel_grid["voxels"]
        input_dict["coordinates"] = voxel_grid["coordinates"]
        input_dict["num_points"] = voxel_grid["num_points_per_voxel"]
        input_dict["voxel_centers"] = (input_dict["coordinates"][:, ::-1] + 0.5) * self.voxel_generator.voxel_size \
                        + self.voxel_generator.point_cloud_range[0:3]
        device = torch.cuda.current_device()

        input_dict["voxels"] = torch.tensor(input_dict["voxels"],dtype=torch.float32,device=device)
        input_dict["coordinates"] = torch.tensor(input_dict["coordinates"], dtype=torch.int32, device=device)
        input_dict["num_points"] = torch.tensor(input_dict["num_points"], dtype=torch.int32, device=device)
        input_dict["voxel_centers"] = torch.tensor(input_dict["voxel_centers"] , dtype=torch.float32, device=device)
        input_dict["image_shape"] = self.image_shape
        zeros_tensor = torch.zeros((input_dict["coordinates"].shape[0],1),dtype=torch.int32,device=device)
        input_dict["coordinates"] = torch.cat([zeros_tensor,input_dict["coordinates"]],dim=1)
        with torch.set_grad_enabled(False):
            input_dict["points_mean"] = input_dict["voxels"][:, :, :].sum(dim=1, keepdim=False)\
                                    / input_dict["num_points"].type_as(input_dict["voxels"] ).view(-1, 1) #vfe
            input_dict["input_sp_tensor"] = spconv.SparseConvTensor(
                features=input_dict["points_mean"],
                indices=input_dict["coordinates"],
                spatial_shape=self.sparse_shape,
                batch_size=1
            )
        input_dict["points"] = mask_points_by_range(points, self.voxel_generator.point_cloud_range)
        input_dict["points"] = torch.tensor(input_dict["points"], dtype=torch.float32, device=device)

        return input_dict

def parpare_point_cloud():
    path = "/media/liang/aabbf09e-0a49-40b7-a5a8-15148073b5d7/liang/kitti/testing/velodyne/000000.bin"
    points = np.fromfile(path,dtype=np.float32).reshape(-1,4)
    return points

def get_fov_flag(points, img_shape, calib):

    # 过滤得到前边90度范围内的点云
    # Valid point should be in the image (and in the PC_AREA_SCOPE)
    # :param pts_rect:
    # :param img_shape:
    # :return:
    pts_rect = calib.lidar_to_rect(points[:, 0:3])
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag

def detect_test():
    prepocess_model = PrepocessData()
    model = torch.load("DetNet.pkl")
    points = parpare_point_cloud()
    fov_flag = get_fov_flag(points, prepocess_model.image_shape, prepocess_model.calib_data)
    points = points[fov_flag]
    with torch.set_grad_enabled(False):
        input_sp_tensor = prepocess_model.points2voxel(points)
        model.cuda()
        model.eval()
        output = model(input_sp_tensor)
    print(output)

class Detection(object):
    def __init__(self):
        super().__init__()
        # self.model = torch.load("DetNet.pkl")
        self.model = DetNet(4)
        self.prepocess_model = PrepocessData()
        self.pub_marker = rospy.Publisher("/object_by_lidar_pvrcnn", MarkerArray,latch=True, queue_size=1)
        self.markers_obj = MarkerArray()
        self.max_marker_size = 0
        self.max_marker_text_size = 0
        code_dir = str(Path(__file__).resolve().parent)
        logger_dir = os.path.join(code_dir,"logger")
        os.makedirs(logger_dir,exist_ok=True)
        log_file = os.path.join(logger_dir,"log_eval_%s.txt" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.logger = common_utils.create_logger(log_file)
        self.ckpt_file = os.path.join(code_dir,"kitti_model.pth")
        # self.ckpt_file = "/home/liang/topic_ws/src/object_detect/for_ros/checkpoint_epoch_80.pth"
        self.model.load_params_from_file(self.ckpt_file,logger=self.logger)
        self.model.cuda()
        self.model.eval()
        print("load model")
        self.output = {}
        self.pub = rospy.Publisher("/object_by_lidar_pvrcnn", MarkerArray,latch=True, queue_size=1)
        self.pub_text = rospy.Publisher("/object_by_lidar_pvrcnn_text", MarkerArray,latch=True, queue_size=1)
    def detect(self,points):
        start=time.time()
        fov_flag = get_fov_flag(points, self.prepocess_model.image_shape, self.prepocess_model.calib_data)
        points = points[fov_flag]
        print("cut_front_point spend time :",time.time()-start)
        # pub_velo = rospy.Publisher("after_cut_velodyne_points", PointCloud2, queue_size=2)
        # publish_pc(points, 'velodyne',pub_velo)
        with torch.set_grad_enabled(False):
            start_voxel=time.time()
            inputs = self.prepocess_model.points2voxel(points)
            print("point to voxel spend time :",time.time()-start_voxel)
            start = time.time()
            if self.output is None:
                self.output.clear()
            self.output.update(self.model(inputs))
            spend_time = time.time() - start
            print("net interfer time :",spend_time)
            print("total detect number:",self.output["box_corner"].shape[0])
        return 0

def publish_result(detection_model):
    boxes3d = detection_model.output["box_corner"]
    labels = detection_model.output["class"]
    boxes_centers = detection_model.output["box_center"]
    # print(oput)
    # pub = rospy.Publisher("/object_by_lidar_pvrcnn", MarkerArray,latch=True, queue_size=1)
    markers_obj = MarkerArray()
    start_time = time.time()
    frame_id ="velo_link"
    for i in range(boxes3d.shape[0]):
        # if max(boxes3d[i][...,2])<-3:
        #     continue
        marker = Marker()
        #指定Marker的参考框架
        marker.header.frame_id = frame_id
        #时间戳
        marker.header.stamp = rospy.Time.now()
        marker.header.seq = i
        #ns代表namespace，命名空间可以避免重复名字引起的错误
        marker.ns = "object_namespace"
        #Marker的id号
        marker.id = i
        #Marker的类型，有ARROW，CUBE等
        marker.type = marker.LINE_STRIP
        #Marker的尺寸，单位是m

        #Marker的动作类型有ADD，DELETE等
        # marker.action = Marker.ADD
        #Marker的位置姿态
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        #Marker的颜色和透明度
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1

        p = Point()
        box = boxes3d[i]
        #add pnts
        p.x = box[0][0]
        p.y = box[0][1]
        p.z = box[0][2]
        marker.points.append(p)
        p = Point()
        p.x = box[1][0]
        p.y = box[1][1]
        p.z = box[1][2]
        marker.points.append(p)
        p = Point()
        p.x = box[2][0]
        p.y = box[2][1]
        p.z = box[2][2]
        marker.points.append(p)
        p = Point()
        p.x = box[3][0]
        p.y = box[3][1]
        p.z = box[3][2]
        marker.points.append(p)
        p = Point()
        p.x = box[0][0]
        p.y = box[0][1]
        p.z = box[0][2]
        marker.points.append(p)
        p = Point()
        p.x = box[4][0]
        p.y = box[4][1]
        p.z = box[4][2]
        marker.points.append(p)
        p = Point()
        p.x = box[5][0]
        p.y = box[5][1]
        p.z = box[5][2]
        marker.points.append(p)
        p = Point()
        p.x = box[1][0]
        p.y = box[1][1]
        p.z = box[1][2]
        marker.points.append(p)
        p = Point()
        p.x = box[5][0]
        p.y = box[5][1]
        p.z = box[5][2]
        marker.points.append(p)
        p = Point()
        p.x = box[6][0]
        p.y = box[6][1]
        p.z = box[6][2]
        marker.points.append(p)
        p = Point()
        p.x = box[2][0]
        p.y = box[2][1]
        p.z = box[2][2]
        marker.points.append(p)
        p = Point()
        p.x = box[6][0]
        p.y = box[6][1]
        p.z = box[6][2]
        marker.points.append(p)
        p = Point()
        p.x = box[7][0]
        p.y = box[7][1]
        p.z = box[7][2]
        marker.points.append(p)
        p = Point()
        p.x = box[3][0]
        p.y = box[3][1]
        p.z = box[3][2]
        marker.points.append(p)
        p = Point()
        p.x = box[7][0]
        p.y = box[7][1]
        p.z = box[7][2]
        marker.points.append(p)
        p = Point()
        p.x = box[4][0]
        p.y = box[4][1]
        p.z = box[4][2]
        marker.points.append(p)
        p = Point()
        p.x = box[0][0]
        p.y = box[0][1]
        p.z = box[0][2]
        marker.points.append(p)

        # detection_model.markers_obj.markers.append(marker)
        markers_obj.markers.append(marker)

        #Marker被自动销毁之前的存活时间，rospy.Duration()意味着在程序结束之前一直存在
        # once
    # detection_model.pub_marker.publish(detection_model.markers_obj)
    # detection_model.pub_marker.publish(markers_obj)
    if len(markers_obj.markers)>detection_model.max_marker_size:
        detection_model.max_marker_size = len(markers_obj.markers)
    if len(markers_obj.markers) !=0:
        for i in range(len(markers_obj.markers),detection_model.max_marker_size+1):

            marker = Marker()
            #指定Marker的参考框架
            marker.header.frame_id = frame_id
            #时间戳
            marker.header.stamp = rospy.Time.now()
            marker.header.seq = i
            #ns代表namespace，命名空间可以避免重复名字引起的错误
            marker.ns = "object_namespace"
            #Marker的id号
            marker.id = i
            #Marker的类型，有ARROW，CUBE等
            marker.type = marker.LINE_STRIP
            marker.color.a = 0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.position.x = 0.0
            marker.pose.position.y = 0.0
            marker.pose.position.z = 0.0
            marker.scale.x = 0.05
            markers_obj.markers.append(marker)

    markers_text = MarkerArray()

    for i in range(boxes3d.shape[0]):
        boxes_center = boxes_centers[i]
        label = labels[i]

        marker_text = Marker()
        marker_text.header.frame_id = frame_id
        marker_text.header.stamp = rospy.Time.now()
        marker_text.header.seq = i + 500
        marker_text.ns = "obj_speed"
        marker_text.id = i + 500
        marker_text.action = Marker.ADD
        marker_text.type = Marker.TEXT_VIEW_FACING
        marker_text.pose.position.x = boxes_center[0]
        marker_text.pose.position.y = boxes_center[1]
        marker_text.pose.position.z = boxes_center[2]

        marker_text.pose.orientation.x = 0
        marker_text.pose.orientation.y = 0
        marker_text.pose.orientation.z = 0
        marker_text.pose.orientation.w = 1
        marker_text.scale.x = 3
        marker_text.scale.y = 3
        marker_text.scale.z = 3
        marker_text.color.r = 1
        marker_text.color.g = 0
        marker_text.color.b = 0
        marker_text.color.a = 1

        # marker_text.text = str(speed)+ ":" + str(label)
        marker_text.text = str(label)
        markers_text.markers.append(marker_text)

    if len(markers_text.markers)>detection_model.max_marker_text_size:
        detection_model.max_marker_text_size = len(markers_text.markers)
    if len(markers_text.markers) !=0:
        for i in range(len(markers_text.markers),detection_model.max_marker_text_size+1):
            marker_text = Marker()
            marker_text.header.frame_id = frame_id
            marker_text.header.stamp = rospy.Time.now()
            marker_text.header.seq = i +500
            marker_text.ns = "obj_speed"
            marker_text.id = i + 500
            marker_text.action = Marker.ADD
            marker_text.type = Marker.TEXT_VIEW_FACING
            marker_text.pose.position.x = 0
            marker_text.pose.position.y = 0
            marker_text.pose.position.z = 0

            marker_text.pose.orientation.x = 0
            marker_text.pose.orientation.y = 0
            marker_text.pose.orientation.z = 0
            marker_text.pose.orientation.w = 1
            marker_text.scale.x = 0.05
            marker_text.scale.y = 0.05
            marker_text.scale.z = 0.05
            marker_text.color.r = 0
            marker_text.color.g = 1
            marker_text.color.b = 0
            marker_text.color.a = 0

            marker_text.text = ""
            markers_text.markers.append(marker_text)

    detection_model.pub_text.publish(markers_text)
    markers_text.markers.clear()

    detection_model.pub.publish(markers_obj)
    # detection_model.markers_obj.markers.clear()
    markers_obj.markers.clear()


    # detection_model.markers_obj.markers.clear(boxes3d.shape[0])
    print("publish object spend time:",time.time() - start_time)
    #    循环发布
    # rate = rospy.Rate(10)
    # while not rospy.is_shutdown():
    #
    #     #发布Marker
    #     pub_maker.publish(marker)
    #
    #     #控制发布频率
    #     rate.sleep()



def velo_callback(msg,detection_model):
    # arr_bbox = BoundingBoxArray()
    start = time.time()
    initial_time = time.time()
    detection_model.markers_obj.markers.clear()

    if len(detection_model.markers_obj.markers)!=0:
        print("not complete clear")
    # print(detection_model.output)
    pc_data = point_cloud2.pointcloud2_to_array(msg)
    points=structured_to_unstructured(pc_data)
    points=points.reshape(-1,4)
    # points = np.array(list(pc_data),dtype = np.float32)
    spend_time = time.time() - start
    print("single subscribe time :",spend_time)
    start = time.time()
    detection_model.detect(points) # start interfer
    spend_time = time.time() - start
    print("detetction time :",spend_time)
    publish_result(detection_model)
    print("total time:", time.time()-initial_time)

    # rospy.init_node('Object_list')
    # pub_velo = rospy.Publisher("/object_by_lidar", Marker, queue_size=1)
    # rospy.loginfo("Initializing...")


def subscribe_point_cloud():
    rospy.init_node('l3Dnet_node')
    detection_model = Detection()
    points = rospy.Subscriber("velodyne_points", PointCloud2,
                             velo_callback,callback_args=detection_model, queue_size=1)
    rospy.spin()
def sub_sequential_lidar():
    rospy.init_node('PVRCNN')
    detection_model = Detection()
    points = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2,
                              velo_callback,callback_args=detection_model, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
    # subscribe_point_cloud()
    sub_sequential_lidar()