from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

import numpy as np
import threading

from visualization_msgs.msg import Marker

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped, Pose
from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy

assert rclpy

from tf_transformations import euler_from_quaternion

import time

from scipy.stats import circmean


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")
        self.lock = threading.Lock()

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        self.particle_publisher = self.create_publisher(PoseArray, "/particle_poses", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        self.N=1000 # Number of particles

        self.particles=np.array([])

        self.cur_time=None

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def get_2Dpose_from_3Dpose(self, pose):
        
        position=pose.position
        quaternion=pose.orientation

        yaw = euler_from_quaternion([quaternion.x,quaternion.y,quaternion.z,quaternion.w])[2]

        return np.array([position.x,position.y,yaw])

    def odom_callback(self, odom_msg):

        if self.cur_time==None:
            self.cur_time=time.time()
            return

        if len(self.particles)==0:
            return
        
        self.lock.acquire()

        v = odom_msg.twist.twist.linear.x
        dtheta = odom_msg.twist.twist.angular.z
        dt = (time.time()-self.cur_time)
        self.cur_time=time.time()

        self.particles = self.motion_model.evaluate(self.particles,np.array([v,dtheta,dt]))
        self.lock.release()

        self.publish_particles()

    def laser_callback(self, laser_msg):

        if len(self.particles)==0:
            return
        
        ranges = np.array(laser_msg.ranges)[::len(laser_msg.ranges)//100]
        
        self.probabilities = self.sensor_model.evaluate(self.particles, ranges)**(1/4)
        self.probabilities/=sum(self.probabilities)

        self.lock.acquire()
        self.particles = self.particles[np.random.choice(self.N, self.N, True, self.probabilities)]
        self.lock.release()

        self.publish_particles()

    def pose_callback(self, pose_msg):

        noise=np.random.multivariate_normal(np.zeros(3),np.eye(3)/10,self.N)

        init_point=self.get_2Dpose_from_3Dpose(pose_msg.pose.pose)

        self.particles=init_point+noise

        self.publish_particles()


    def yaw_to_quaternion(self, yaw):
    # Convert yaw angle to quaternion
        quaternion = np.zeros(4)
        quaternion[2] = np.sin(yaw / 2)
        quaternion[3] = np.cos(yaw / 2)
        return quaternion

    def publish_particles(self):
        pose_array = PoseArray()
        pose_array.header.frame_id = "map"
        pose_array.header.stamp = self.get_clock().now().to_msg()

        for particle in self.particles:
            pose = Pose()
            pose.position.x = particle[0]
            pose.position.y = particle[1]
            quaternion = self.yaw_to_quaternion(particle[2])
            pose.position.z = 0.0
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]
            pose_array.poses.append(pose)

        odom_msg=Odometry()
        odom_msg.header.frame_id="map"
        odom_msg.header.stamp=self.get_clock().now().to_msg()
        odom_msg.pose.pose.position.x=np.mean(self.particles[:,0])
        odom_msg.pose.pose.position.y=np.mean(self.particles[:,1])
        angle=circmean(self.particles[:,2])
        quaternion=self.yaw_to_quaternion(angle)
        odom_msg.pose.pose.orientation.x=quaternion[0]
        odom_msg.pose.pose.orientation.y=quaternion[1]
        odom_msg.pose.pose.orientation.z=quaternion[2]
        odom_msg.pose.pose.orientation.w=quaternion[3]

        self.particle_publisher.publish(pose_array)
        self.odom_pub.publish(odom_msg)



def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()