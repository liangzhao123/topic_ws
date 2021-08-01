import rospy

#导入显示消息类型，这里主要是Marker消息类型
from visualization_msgs.msg import *

#main
if __name__=="__main__":

    #初始化节点，anonymous为True，避免重复名字
    rospy.init_node("cube", anonymous=True)

    #发布频率
    rate = rospy.Rate(10)

    #定义发布者，发布话题：/cube，消息类型是：Marker,消息队列长度：10
    marker_pub = rospy.Publisher("/cube", Marker, queue_size=10)

    rospy.loginfo("Initializing...")

    #定义一个marker对象，并初始化各种Marker的特性
    marker = Marker()

    #指定Marker的参考框架
    marker.header.frame_id = "velodyne"

    #时间戳
    marker.header.stamp = rospy.Time.now()

    #ns代表namespace，命名空间可以避免重复名字引起的错误
    marker.ns = "basic_shapes"

    #Marker的id号
    marker.id = 0

    #Marker的类型，有ARROW，CUBE等
    marker.type = Marker.CYLINDER

    #Marker的尺寸，单位是m
    marker.scale.x = 1
    marker.scale.y = 2
    marker.scale.z = 1

    #Marker的动作类型有ADD，DELETE等
    marker.action = Marker.ADD

    #Marker的位置姿态
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.2
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    #Marker的颜色和透明度
    marker.color.r = 0.0
    marker.color.g = 0.8
    marker.color.b = 0.0
    marker.color.a = 0.5

    #Marker被自动销毁之前的存活时间，rospy.Duration()意味着在程序结束之前一直存在
    marker.lifetime = rospy.Duration()
    #循环发布
    while not rospy.is_shutdown():
        #发布Marker
        marker_pub.publish(marker)
        print("a")
        #控制发布频率
        rate.sleep()