#!/usr/bin/env python3
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from object_localizer.yolov3_tf2.models import (YoloV3, YoloV3Tiny)
from object_localizer.yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from object_localizer.yolov3_tf2.utils import draw_outputs
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CompressedImage
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose, PoseArray
import rospkg 
from visualization_msgs.msg import Marker , MarkerArray
from object_localizer import __path__ as local_path
from object_msgs.msg import Object, ObjectArray

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def get_pose_from_bbox(bbox, pcl_msg, image_shape):
    xmin, ymin, xmax, ymax = bbox
    original_height = image_shape[0]
    original_width = image_shape[1]
    xm = int((xmin + xmax)/2)
    ym = int((ymin + ymax)/2)
    if xm >= original_width:
        xm = original_width - 1
    if ym >= original_height:
        ym = original_height - 1
    data_out = pc2.read_points(pcl_msg, field_names=("x", "y", "z"), skip_nans=False, uvs=[[xm,ym]])
    point = next(data_out)
    return point

def new_marker(name, id, pose, frame_id):
    marker_name = Marker()
    marker_name.header.frame_id = frame_id
    marker_name.header.stamp = rospy.Time.now()
    # Orientation of the text
    marker_name.pose.orientation.x = 0
    marker_name.pose.orientation.y = 0
    marker_name.pose.orientation.z = 0
    marker_name.pose.orientation.w = 1
    # Colore of text
    marker_name.color.r = 0
    marker_name.color.g = 0
    marker_name.color.b = 1
    marker_name.color.a = 2.0
    # Rest of things:
    marker_name.id = id
    marker_name.type = 9
    marker_name.action = 0
    marker_name.lifetime.secs = 1
    marker_name.pose.position.x = pose[0]
    marker_name.pose.position.y = pose[1]
    marker_name.pose.position.z = pose[2]
    # Size of the text
    marker_name.scale.x = 0.1
    marker_name.scale.y = 0.1
    marker_name.scale.z = 0.1
    marker_name.text = name
    marker_name.ns = name + str(id)
    return marker_name
    


class YoloDetectorNode():
    def __init__(self,
                 frame_id,
                 image_topic,
                 image_topic_debug,
                 objects_topic,
                 point_cloud_topic,
                 objects_markers_topic,
                 loop_duration = 1,
                 publish_markers = True,
                 publish_marked_image = True,
                 tiny = False):
        print("Initializing yolo 3 detector....")
        self.frame_id = frame_id
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for physical_device in physical_devices:
                tf.config.experimental.set_memory_growth(physical_device, True)
        if tiny == True:
            self.yolo = YoloV3Tiny(classes=80)
            self.yolo.load_weights(local_path[0]+'/checkpoints/yolov3-tiny.tf').expect_partial()
        else:
            self.yolo = YoloV3(classes=80)
            self.yolo.load_weights(local_path[0]+'/checkpoints/yolov3.tf').expect_partial()
        print('Weights loaded!')
        self.class_names = [c.strip() for c in open(local_path[0]+'/data/coco.names').readlines()]
        print("Classes loaded!")
        rospy.Timer(rospy.Duration(loop_duration), self.img_callback)
        rospy.Subscriber(image_topic, CompressedImage, self._img_callback, queue_size=1)
        self.point_cloud_topic = point_cloud_topic
        self.image_topic = image_topic
        self.publish_markers = publish_markers
        self.publish_marked_image = publish_marked_image
        self.objects_publisher = rospy.Publisher(objects_topic, ObjectArray, queue_size=1)  
        if self.publish_markers:
            self.objects_markers_publisher = rospy.Publisher(objects_markers_topic , MarkerArray , queue_size = 1)
        if self.publish_marked_image:
            self.image_publisher = rospy.Publisher(image_topic_debug, CompressedImage, queue_size=1)
        rospy.on_shutdown(self.shutdown_callback)
        self.img = None
        print("Initialization done!")
    def shutdown_callback(self):
        cv2.destroyAllWindows() 

    def _img_callback(self, ros_img):
        ros_img = rospy.wait_for_message(self.image_topic, CompressedImage)
        img = np.fromstring(ros_img.data, np.uint8)
        self.img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    def img_callback (self, data):
        if self.img is not None:
            img = self.img.copy()
            raw_img = self.img.copy()
            objects_msg = ObjectArray()
            img = tf.expand_dims(img, 0)
            img = transform_images(img, 416)
            boxes, scores, classes, nums, probabilities = self.yolo(img)
            np_boxes = boxes.numpy()
            pcl_msg = rospy.wait_for_message(self.point_cloud_topic, PointCloud2)

            list_marker = MarkerArray()
            list_marker.markers = []
            for i in range(nums.numpy()[0]):
                object_msg = Object()
                object_msg.header.stamp = rospy.Time.now()
                object_msg.header.frame_id = self.frame_id

                ymin = np.int32(np_boxes[0][i][1] * raw_img.shape[0])
                xmin = np.int32(np_boxes[0][i][0] * raw_img.shape[1])
                ymax = np.int32(np_boxes[0][i][3] * raw_img.shape[0])
                xmax = np.int32(np_boxes[0][i][2] * raw_img.shape[1])
                object_msg.bounds = [xmin, ymin, xmax, ymax]
                object_msg.cls = self.class_names[np.array(classes[0][i]).astype(np.int32)]
                object_msg.probability = np.array(scores[0][i])
                point = get_pose_from_bbox(object_msg.bounds, pcl_msg, raw_img.shape)
                if np.sum(np.isnan(point)) == 0:
                    if np.linalg.norm(point) < 5:
                        object_msg.is_pose_nan = False
                        object_msg.pose.x = point[0]
                        object_msg.pose.y = point[1]
                        object_msg.pose.z = point[2]
                        if self.publish_markers:
                            marker_temp = new_marker(self.class_names[np.array(classes[0][i]).astype(np.int32)], i, point, self.frame_id)
                            list_marker.markers.append(marker_temp)
                        objects_msg.objects.append(object_msg)
            if len(objects_msg.objects) > 0:
                self.objects_publisher.publish(objects_msg)
                if self.publish_markers:
                    self.objects_markers_publisher.publish(list_marker)
            
            if self.publish_marked_image:
                out_img = draw_outputs(raw_img, (boxes, scores, classes, nums), self.class_names)
                out_img = out_img.astype(np.uint8)
                msg = CompressedImage()
                msg.header.stamp = rospy.Time.now()
                msg.format = "jpeg"
                msg.header.frame_id = 'camera_link'
                msg.header.stamp = rospy.Time.now()
                compressed_img = np.array(cv2.imencode('.jpg', out_img)[1])
                msg.data = compressed_img.tostring()
                self.image_publisher.publish(msg)






if __name__ == '__main__':
    rospy.init_node('object_localizer_node', anonymous = True)
    image_topic = rospy.get_param("/object_detector/image_topic", '/camera/color/image_raw/compressed')
    image_topic_debug = rospy.get_param("/object_detector/image_topic_debug", '/object_detector/image_raw/compressed')
    objects_topic = rospy.get_param("/object_detector/objects_topic", '/object_detector/objects')
    point_cloud_topic = rospy.get_param("/object_detector/point_cloud_topic", '/camera/depth/points')
    objects_markers_topic = rospy.get_param("/object_detector/objects_markers_topic", '/object_detector/objects/markers')
    loop_duration = rospy.get_param("/object_detector/loop_duration", 1)
    publish_markers = rospy.get_param("/object_detector/publish_markers", True)
    publish_marked_image = rospy.get_param("/object_detector/publish_marked_image", True)
    frame_id = rospy.get_param("/object_detector/frame_id", 'camera_link')
    detector = YoloDetectorNode(frame_id,
                                image_topic,
                                image_topic_debug,
                                objects_topic,
                                point_cloud_topic,
                                objects_markers_topic,
                                loop_duration,
                                publish_markers,
                                publish_marked_image)

    rospy.spin()
    rospy.signal_shutdown("exit_detector_node shutdown...")



    
  

    
