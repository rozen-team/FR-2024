#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import math

from enum import Enum
from typing import Tuple, Optional

from clover import srv
from std_srvs.srv import Trigger
from sensor_msgs.msg import Range

from detector import *

class VideoSource:
    class SourceType(Enum):
        File = 0
        Topic = 1

    def __init__(self, callback, bridge: CvBridge, source: str = "/main_camera/image_raw_throttled", type: SourceType = SourceType.Topic):
        self.source = source
        self.type = type
        self.callback = callback
        self.bridge = bridge

        if self.type == self.SourceType.File:
            # Для тестирования кода (с использованием видео)
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap:
                raise Exception(f"Couldn`t open video file {self.source}")
            else:
                print(f"Successfully open video file {self.source}")

            # Тестовые значения
            self.cm = np.array([[ 92.37552066, 0., 160.5], [0., 92.37552066, 120.5], [0., 0., 1.]], dtype="float64")
            self.dc = np.zeros(5, dtype="float64")

            rospy.Timer(rospy.Duration(1/30), self.video_callback)
        else:
            self.cm, self.dc = self.camera_cfg_cvt(rospy.wait_for_message("/main_camera/camera_info", CameraInfo))
            self.image_sub = rospy.Subscriber(self.source, Image, self.callback)

    def camera_cfg_cvt(self, msg: CameraInfo) -> Tuple[np.ndarray, np.ndarray]:
        return (np.reshape(np.array(msg.K, dtype="float64"), (3, 3)), np.array(msg.D, dtype="float64"))

    # Метод, реализующий публикацию видеопотока (используется для тестирование кода)
    def video_callback(self, event):
        ret, frame = self.cap.read()

        if not ret:
            raise Exception("End of video file")

        self.callback(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

class NodeHandle:
    # Пороговые значения для пола
    floor_thr = [
        np.array([0, 0, 0]),
        np.array([180, 255, 120])
    ]

    FLIGHT_HEIGHT = 1.2

    def __init__(self, source: str = "/main_camera/image_raw_throttled", type: VideoSource.SourceType = VideoSource.SourceType.Topic):
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
        self.set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
        self.set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
        self.set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
        self.set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
        self.land = rospy.ServiceProxy('land', Trigger)

        self.bridge = CvBridge()
        self.is_start = False

        self.floor_mask_pub = rospy.Publisher("/a/floor_mask", Image, queue_size=1)
        self.path_pub = rospy.Publisher("/a/path_points", MarkerArray, queue_size=1)

        self.video_source = VideoSource(callback=self.callback, source=source, type=type, bridge=self.bridge)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.objects_searcher = ObjectSearcher(cm=self.video_source.cm, dc=self.video_source.dc, tf_buffer=self.tf_buffer, cv_bridge=self.bridge)

    def takeoff(self, use_height=False, takeoff_thr=0.1):
        dist = 0.0

        if use_height:
            dist = rospy.wait_for_message('rangefinder/range', Range).range
        
        if abs(dist) <= max(takeoff_thr, 0.1):
            self.navigate_wait(z=self.FLIGHT_HEIGHT, speed = 0.6, frame_id='body', auto_arm=True)
            rospy.sleep(2)
            print(f'Successfully takeoff!')

            return

        print(f'Takeof error: clover is already takeoff!')

    def navigate_wait(self, x=0, y=0, z=0, yaw=float('nan'), speed=0.5, frame_id='', auto_arm=False, tolerance=0.15):
        self.navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

        while not rospy.is_shutdown():
            telem = self.get_telemetry(frame_id='navigate_target')
            if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
                break
            rospy.sleep(0.2)

    def enable(self):
        self.is_start = True

    def disable(self):
        self.is_start = False

    # Метод, создающий маску для полаы
    def floor_mask(self, hsv):
        hsv = cv2.blur(hsv, (10, 10))
        mask = cv2.inRange(hsv, self.floor_thr[0], self.floor_thr[1])

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        mask = np.zeros(mask.shape, dtype="uint8")
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
            
            area = cv2.contourArea(approx)
            if area < 600:
                continue
        
            mask = cv2.fillPoly(mask, pts = [approx], color=(255,255,255))
        
        return mask

    # Callback-метод топика с изображением
    def callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        if not self.is_start:
            return

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Создаем маску для пола площадки
        floor_mask = self.floor_mask(hsv)
        self.floor_mask_pub.publish(self.bridge.cv2_to_imgmsg(floor_mask, "mono8"))

        self.objects_searcher.on_frame(image, mask_floor=floor_mask, hsv=hsv)

def main():
    rospy.init_node('first_task', anonymous=True)
    
    try:
        #node = NodeHandle(source="output_1.avi", type=VideoSource.SourceType.File)
        node = NodeHandle()

        node.enable()
        node.takeoff(use_height = True)

        node.navigate_wait(x=1, y=2.5, z=node.FLIGHT_HEIGHT, speed=0.3, frame_id="aruco_map")

        rospy.spin()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()