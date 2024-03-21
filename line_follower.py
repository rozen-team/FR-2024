import rospy
from clover import srv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Range

import cv2

from enum import Enum
from math import fmod, pi

import numpy as np

import time

class LineFollower:
    class States(Enum):
        Init = 0
        Reverse = 1
        Follow = 2
        End = 3

    def __init__(self, 
                 line_threshold,
                 bridge,
                 target_height = 1.0, 
                 debug_publisher = True,
                 k_velocity_y = -0.006,
                 line_velocity = 0.08,
                 k_velocity_z = 0.04):
        self.state = self.States.Init
        self.debug_publisher = debug_publisher
        self.bridge = bridge
        self.blur_kernel = (5, 5)
        self.morph_kernel = np.ones((5, 5), np.uint8)

        assert len(line_threshold) == 2
        self.threshold = line_threshold

        # коэффициент движения по оси Y за линией
        self.k_velocity_y = k_velocity_y
        # скорость движения за линией
        self.line_velocity = line_velocity
        # коэффициент движения по оси Z
        self.k_velocity_z = k_velocity_z

        self.y_roi = (120, 30)
        self.min_area = 300

        self.target_yaw = None
        self.target_height = target_height

        self.end_time = 3.0 # sec
        self.last_line_seen = None

        self.is_enable = False

        self.height = self.target_height

        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)

        self.range_sub = rospy.Subscriber('rangefinder/range', Range, self.range_cb)

        if self.debug_publisher:
            self.debug_pub = rospy.Publisher("/a/line_debug", Image, queue_size=1)

    def range_cb(self, msg):
        self.height = msg.range

    def ang_norm(self, ang):
        ang = fmod(fmod(ang, 2.0 * pi) + 2.0 * pi, 2.0 * pi)
        if ang > pi:
            ang -= 2.0 * pi
        return ang

    def enable(self):
        self.is_enable = True

    def disable(self):
        self.is_enable = False

    def update(self, image: np.ndarray, hsv: np.ndarray):
        debug = self.on_frame(image, hsv)
        if debug is not None:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, "bgr8"))

    # распознавание линии и повреждений
    def on_frame(self, image: np.ndarray, hsv: np.ndarray):
        debug = None
        if self.debug_publisher: debug = image.copy()

        if not self.is_enable:
            return debug

        if self.state == self.States.End:
            return debug

        if self.state == self.States.Reverse:
            pose = self.get_telemetry(frame_id = 'aruco_map')
            ang_min = self.ang_norm(self.reverse_yaw - 0.15)
            ang_max = self.ang_norm(self.reverse_yaw + 0.15)

            if pose.yaw >= ang_min and pose.yaw <= ang_max:
                self.state = self.States.Follow
            else:
                return debug

        height, width, _ = image.shape

        # бинаризуем изображение из пространства HSV
        # в этом пространстве легче выделить желтый цвет
        hsv_blur = cv2.GaussianBlur(hsv, self.blur_kernel, 0)
        binary = cv2.inRange(hsv, self.threshold[0], self.threshold[1])

        binary = binary[(height // 2) - self.y_roi[0]:(height // 2) + self.y_roi[1], :]
        binary = cv2.erode(binary, self.morph_kernel)
        binary = cv2.dilate(binary, self.morph_kernel)

        if self.debug_publisher:
            debug = cv2.rectangle(debug, (0, (height // 2) - self.y_roi[0]), (width, (height // 2) + self.y_roi[1]), (0, 255, 0), 2)

        # ищем контуры линии
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(filter(lambda c: cv2.contourArea(c) > self.min_area, contours))

        major_contour = None
        if len(contours) != 0: major_contour = min(contours, key = lambda c: np.int0(cv2.boxPoints(cv2.minAreaRect(c)))[0][1])

        if major_contour is not None:
            rect = cv2.minAreaRect(major_contour)
            (x_min, y_min), (w_min, h_min), _ = rect

            if self.debug_publisher:
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                box = [[p[0], p[1] + (height // 2) - self.y_roi[0]] for p in box]
                box = np.array(box)
                debug = cv2.drawContours(debug, [box], 0, (255, 120, 0), 2)
                debug = cv2.circle(debug, (int(x_min), int(y_min + (height // 2) - self.y_roi[0])), 5, (255, 120, 0), -1)

            y_min += (height // 2) - self.y_roi[0]
            thr_low = (height // 2) - 15

            if y_min >= thr_low and self.state == self.States.Init:
                pose = self.get_telemetry(frame_id = 'aruco_map')
                self.target_yaw = self.ang_norm(pose.yaw + pi)

                print(f'Reverse clover to follow the line...')

                self.navigate(
                    x = pose.x, 
                    y = pose.y, 
                    z = pose.z,
                    yaw = self.target_yaw, 
                    frame_id='aruco_map'
                )
                self.target_yaw = need_yaw
                self.state = self.States.Reverse

            else:
                self.state = self.States.Follow
                error = x_min - (width / 2)

                self.set_velocity(
                    vx = self.line_velocity,
                    vy = error * self.k_velocity_y,
                    vz = (self.target_height - self.height) * self.k_velocity_z,
                    yaw = float('nan'), 
                    frame_id = 'body'
                )

            self.last_line_seen = None

        else:
            now = time.time()
            if self.last_line_seen is None: 
                self.last_line_seen = now
            elif (now - self.last_line_seen) >= self.end_time:
                self.state = self.States.End
                self.set_velocity(
                    vx = 0.0, 
                    vy = 0.0, 
                    vz = 0.0,
                    yaw = float('nan'), 
                    frame_id = 'body'
                )

        return debug
