import sys
sys.path.append("/home/liang/topic_ws/src/object_detect")
import numpy as np
import spconv
import torch

import torch.nn as nn

import rospy
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header

def parpare_point_cloud():
    path = "/home/liang/kitti/testing/velodyne/000000.bin"
    points = np.fromfile(path,dtype=np.float32).reshape(-1,4)
    return points
class PrepocessData(object):
    def __init__(self):
        super().__init__()
        # self.voxel_generator_cfg = cfg.DATA_CONFIG.VOXEL_GENERATOR
        self.voxel_generator = spconv.utils.VoxelGeneratorV2(
        voxel_size=[0.05,0.05,0.1],
        point_cloud_range=[0,-40.0,-3.0,70.4,40.0,1.0],
        max_num_points=5,
        max_voxels=16000
    )
    def points2voxel(self,points):
        voxel_grid = self.voxel_generator.generate(points)
        voxels = voxel_grid["voxels"]
        coordinates = voxel_grid["coordinates"]
        num_points = voxel_grid["num_points_per_voxel"]
        return voxels, coordinates, num_points

class PostProcess(object):
    def __init__(self):
        super().__init__()
        pass
class DetNet(nn.Module):
    def __init__(self):
        super().__init__()
        pass

def point2voxel(points):
    # voxel_generator_cfg = cfg.DATA_CONFIG.VOXEL_GENERATOR
    voxel_generator = spconv.utils.VoxelGeneratorV2(
        voxel_size=[0.05,0.05,0.1],
        point_cloud_range=[0,-40.0,-3.0,70.4,40.0,1.0],
        max_num_points=5,
        max_voxels=16000
    )
    voxel_grid = voxel_generator.generate(points)
    voxels = voxel_grid["voxels"]
    coordinates = voxel_grid["coordinates"]
    num_points = voxel_grid["num_points_per_voxel"]
    return voxels,coordinates,num_points

def boxes3d_to_corners3d_lidar_torch(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords, see the definition of ry in KITTI dataset
    :param z_bottom: whether z is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    w, l, h = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6]
    ry = boxes3d[:, 6:7]

    zeros = torch.cuda.FloatTensor(boxes_num, 1).fill_(0)
    ones = torch.cuda.FloatTensor(boxes_num, 1).fill_(1)
    x_corners = torch.cat([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dim=1)  # (N, 8)
    y_corners = torch.cat([-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.], dim=1)  # (N, 8)
    if bottom_center:
        z_corners = torch.cat([zeros, zeros, zeros, zeros, h, h, h, h], dim=1)  # (N, 8)
    else:
        z_corners = torch.cat([-h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.], dim=1)  # (N, 8)
    temp_corners = torch.cat((
        x_corners.unsqueeze(dim=2), y_corners.unsqueeze(dim=2), z_corners.unsqueeze(dim=2)
    ), dim=2)  # (N, 8, 3)

    cosa, sina = torch.cos(ry), torch.sin(ry)
    raw_1 = torch.cat([cosa, -sina, zeros], dim=1)  # (N, 3)
    raw_2 = torch.cat([sina,  cosa, zeros], dim=1)  # (N, 3)
    raw_3 = torch.cat([zeros, zeros, ones], dim=1)  # (N, 3)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1), raw_3.unsqueeze(dim=1)), dim=1)  # (N, 3, 3)

    rotated_corners = torch.matmul(temp_corners, R)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]
    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.view(-1, 1) + x_corners.view(-1, 8)
    y = y_loc.view(-1, 1) + y_corners.view(-1, 8)
    z = z_loc.view(-1, 1) + z_corners.view(-1, 8)
    corners = torch.cat((x.view(-1, 8, 1), y.view(-1, 8, 1), z.view(-1, 8, 1)), dim=2)

    return corners


def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]




def publish_pc(np_array, frame_id,pub_velo):
    header = Header()
    header.stamp = rospy.Time()
    header.frame_id = frame_id

    x = np_array[:, 0].reshape(-1)
    y = np_array[:, 1].reshape(-1)
    z = np_array[:, 2].reshape(-1)

    # if intensity field exists
    if np_array.shape[1] == 4:
        i = np_array[:, 3].reshape(-1)
    else:
        i = np.zeros((np_array.shape[0], 1)).reshape(-1)

    cloud = np.stack((x, y, z, i))
    # point cloud segments
    # 4 PointFields as channel description
    msg_segment = pc2.create_cloud(header=header,
                                   fields=_make_point_field(4),
                                   points=cloud.T)

    #  publish to /velodyne_points_modified
    pub_velo.publish(msg_segment)  # DEBUG