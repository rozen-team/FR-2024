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
from buildings_detector import *
import time

from math import pi, fmod

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

    FLIGHT_HEIGHT = 2.0
    LINE_HEIGHT = 1.0

    def __init__(self, source: str = "/main_camera/image_raw_throttled", type: VideoSource.SourceType = VideoSource.SourceType.Topic):
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.navigate_global = rospy.ServiceProxy('navigate_global', srv.NavigateGlobal)
        self.set_position = rospy.ServiceProxy('set_position', srv.SetPosition)
        self.set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
        self.set_attitude = rospy.ServiceProxy('set_attitude', srv.SetAttitude)
        self.set_rates = rospy.ServiceProxy('set_rates', srv.SetRates)
        self.land = rospy.ServiceProxy('land', Trigger)
        self.set_yaw_rate = rospy.ServiceProxy('set_yaw_rate', srv.SetYawRate)

        self.bridge = CvBridge()
        self.is_start = False
        self.is_screenshot_handled = False

        self.line_end = False
        self.is_reversing = False
        self.reverse_yaw = 0.0
        # коэффициент поворота за линией
        self.k_angle = -0.006
        # коэффициент движения по оси Y за линией
        self.k_velocity_y = -0.006
        # скорость движения за линией
        self.line_velocity = 0.097
        self.line_end_thr = 15.0
        # время, которое должно пройти с исчезновения линии для возвращения на точку старта
        self.line_end_thr = 3.0
        self.line_end_time = 0.0
        self.first_reverse = True

        self.is_line_enabled = False


        self.screenshot_name = 'empty'

        self.floor_mask_pub = rospy.Publisher("/a/floor_mask", Image, queue_size=1)
        self.qr_debug_pub = rospy.Publisher("/a/qr_debug", Image, queue_size=1)
        self.line_debug_pub = rospy.Publisher("/a/line_debug", Image, queue_size=1)
        self.path_pub = rospy.Publisher("/a/path_points", MarkerArray, queue_size=1)

        self.video_source = VideoSource(callback=self.callback, source=source, type=type, bridge=self.bridge)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.objects_searcher = ObjectSearcher(cm=self.video_source.cm, dc=self.video_source.dc, tf_buffer=self.tf_buffer, cv_bridge=self.bridge)
        self.buildings_searcher = BuildingsDetector(cm=self.video_source.cm, dc=self.video_source.dc, tf_buffer=self.tf_buffer, cv_bridge=self.bridge)

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

        if self.is_screenshot_handled:
            cv2.imwrite(f'{self.screenshot_name}.png', image)
            self.is_screenshot_handled = False

        line_debug = image.copy()
        if self.is_line_enabled:
            line_debug = self.line_detect(image)

            self.line_debug_pub.publish(self.bridge.cv2_to_imgmsg(line_debug, "bgr8"))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Создаем маску для пола площадки
        floor_mask = self.floor_mask(hsv)
        self.floor_mask_pub.publish(self.bridge.cv2_to_imgmsg(floor_mask, "mono8"))

        self.objects_searcher.on_frame(image, mask_floor=floor_mask, hsv=hsv)
        self.buildings_searcher.on_frame(image)

    def ang_norm(self, ang):
        a = fmod(fmod(ang, 2.0 * pi) + 2.0 * pi, 2.0 * pi)
        if a > pi:
            a -= 2.0 * pi
        return a

    def ang_normilize(self, w_min, h_min, ang):
        if ang < -45:
                ang = 90 + ang
        if w_min < h_min and ang > 0:
                ang = (90 - ang) * -1
        if w_min > h_min and ang < 0:
                ang = 90 + ang
        return ang

    # распознавание линии и повреждений
    def line_detect(self, image):
        debug = image.copy()

        if self.line_end:
            print("Line is end!")
            return debug

        if self.is_reversing:
            pose = self.get_telemetry(frame_id = 'aruco_map')
            ang_min = self.ang_norm(self.reverse_yaw - 0.15)
            ang_max = self.ang_norm(self.reverse_yaw + 0.15)

            if pose.yaw >= ang_min and pose.yaw <= ang_max:
                print("Reverse was finished")
                self.is_reversing = False
            else:
                return debug

        height, width, _ = image.shape

        blur = cv2.GaussianBlur(image, (5,5), 0)

        # бинаризуем изображение из пространства HSV
        # в этом пространстве легче выделить желтый цвет
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        bin = cv2.inRange(hsv, \
            (23, 48, 141), (52, 180, 255))

        bin = bin[(height // 2) - 60:(height // 2) + 30, :]
        kernel = np.ones((5,5),np.uint8)
        bin = cv2.erode(bin, kernel)
        bin = cv2.dilate(bin, kernel)

        debug = cv2.rectangle(debug, (0, (height // 2) - 60), (width, (height // 2) + 30), (0, 255, 0), 2)

        # ищем контуры линии
        contours, _ = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        compute_rect = [float('inf'), (0, 0), (0, 0), 0, (0, 0), (0, 0)]
        center = 0.0
        for cnt in contours:
            # фильтрация по площади в пикселях
            area = cv2.contourArea(cnt)
            if cv2.contourArea(cnt) > 300:
                rect = cv2.minAreaRect(cnt)
                bx, by, bw, bh = cv2.boundingRect(cnt)
                (x_min, y_min), (w_min, h_min), angle = rect

                if True:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    # смещаем точки для корректного отображения
                    box = [[p[0], p[1] + (height // 2) - 60] for p in box]

                    # рисуем
                    box = np.array(box)
                    debug = cv2.drawContours(debug, [box], 0, (255, 120, 0), 2)
                    debug = cv2.circle(debug, (int(x_min), int(y_min + (height // 2) - 60)), 5, (255, 120, 0), -1)

                    # сохраняем контур с максимальной координатой y для последующего расчета скорости клевера
                    if compute_rect[0] > box[0][1]:
                        center = int(x_min)
                        compute_rect = [box[0][1], (x_min, y_min), (w_min, h_min), angle, (bx, by), (bw, bh)]

        # рассчитываем скорости коптера для движения за линией
        if compute_rect[0] != float('inf'):
            _, (x_min, y_min), (w_min, h_min), angle, (bx, by), (bw, bh) = compute_rect
            
            # если линия перевернута на 180 градусов, то поворачиваемся
            angle = self.ang_normilize(w_min, h_min, angle)
            y_min += (height // 2) - 60

            frame_cn = (height / 2)
            thr_low = frame_cn - 15

            if y_min >= thr_low and self.first_reverse:
                pose = self.get_telemetry(frame_id = 'aruco_map')
                need_yaw = self.ang_norm(pose.yaw + pi)

                print(f'Reverse line need_yaw={need_yaw}')

                self.navigate(x = pose.x, y = pose.y, z = pose.z, \
                     yaw = need_yaw, frame_id='aruco_map')
                self.reverse_yaw = need_yaw
                self.is_reversing = True

            else:
                pose = self.get_telemetry(frame_id = 'aruco_map')
                image_draw = cv2.circle(debug, (int(center), int(y_min)), 8, (127, 127, 127), -1)

                self.first_reverse = False
                error = center - (width / 2)

                # (self.LINE_HEIGHT - pose.z) * 0.04
                self.set_velocity(vx = self.line_velocity, vy = error * self.k_velocity_y, vz = (self.LINE_HEIGHT - pose.z) * 0.04, \
                    yaw = float('nan'), frame_id = 'body')
                #self.set_yaw_rate(yaw_rate=angle * self.k_angle)

            
            self.line_end_time = int(self.line_end) * time.time()

        else:
            now = time.time()
            if self.line_end_time == 0 and (not self.line_end): self.line_end_time = now
            elif self.line_end_time > 0.0 and (now - self.line_end_time) >= self.line_end_thr:
                self.line_end = True
                self.set_velocity(vx = 0.0, vy = 0.0, vz = 0.0, \
                    yaw = float('nan'), frame_id = 'body')

        return debug

    def take_screenshot(self, name):
        self.screenshot_name = name
        self.is_screenshot_handled = True
        while self.is_screenshot_handled:
            rospy.sleep(0.1)

        print("Screenshot Successfully taked!")

def main():
    rospy.init_node('first_task', anonymous=True)
    
    BUILDINGS_LEFT_BOTTOM = (4.0, 1.0)
    BUILDINGS_RIGHT_TOP = (7.0, 4.0)
    BUILDINGS_CENTER = (float(BUILDINGS_LEFT_BOTTOM[0] + BUILDINGS_RIGHT_TOP[0]) / 2.0,
                        float(BUILDINGS_LEFT_BOTTOM[1] + BUILDINGS_RIGHT_TOP[1]) / 2.0)

    BUILDINGS_HEIGHT = 2.0
    BUILDINGS_SEARCH_HEIGHT = 1.5
    BUILDINGS_STEP = 1.0
    BUILDINGS_BORDER = 0.2
    BUILDINGS_SEARCH_VEL = 0.3

    LINE_START = (1.2, 0.5)
    #LINE_HEIGHT = 1.0
    LINE_VEL = 0.2

    ENABLE_BUILDINGS = True
    ENABLE_LINE = False


    try:
        #node = NodeHandle(source="output_1.avi", type=VideoSource.SourceType.File)
        node = NodeHandle()

        node.enable()
        node.takeoff(use_height = True)

        if ENABLE_BUILDINGS:
            node.navigate_wait(x=BUILDINGS_CENTER[0], y=BUILDINGS_CENTER[1], z=BUILDINGS_HEIGHT, speed=0.3, frame_id="aruco_map")
            rospy.sleep(2)

            # take screenshot of buildings area
            node.take_screenshot(f'buildings_area')

            # fly by rect to find all buildings
            point = [BUILDINGS_LEFT_BOTTOM[0] + BUILDINGS_BORDER, BUILDINGS_LEFT_BOTTOM[1] + BUILDINGS_BORDER]
            steps = float(BUILDINGS_RIGHT_TOP[1] - BUILDINGS_LEFT_BOTTOM[1] + (2 * BUILDINGS_BORDER)) / BUILDINGS_STEP
            for i in range(int(steps)):
                node.navigate_wait(x=point[0], y=point[1], z=BUILDINGS_SEARCH_HEIGHT, speed=BUILDINGS_SEARCH_VEL, frame_id="aruco_map")

                if i % 2 == 0: node.buildings_searcher.enable()

                if i % 2 == 0:
                    point[0] = BUILDINGS_RIGHT_TOP[0] - BUILDINGS_BORDER
                else:
                    point[0] = BUILDINGS_LEFT_BOTTOM[0] + BUILDINGS_BORDER

                node.navigate_wait(x=point[0], y=point[1], z=BUILDINGS_SEARCH_HEIGHT, speed=BUILDINGS_SEARCH_VEL, frame_id="aruco_map")

                point[1] += BUILDINGS_STEP

            node.buildings_searcher.disable()
            buildings_coords = node.buildings_searcher.send_request()
            building_height = node.buildings_searcher.get_height(buildings_coords)

            node.navigate_wait(x=buildings_coords[0], y=buildings_coords[1], z=BUILDINGS_HEIGHT, speed=BUILDINGS_SEARCH_VEL, frame_id="aruco_map")
            node.navigate_wait(x=buildings_coords[0], y=buildings_coords[1], z=building_height + 0.5, speed=BUILDINGS_SEARCH_VEL, frame_id="aruco_map")

            # take screenshot of building
            node.take_screenshot(f'building_{buildings_coords[0]}_{buildings_coords[0]}')

        if ENABLE_LINE:
            node.navigate_wait(x=LINE_START[0], y=LINE_START[1], z=node.FLIGHT_HEIGHT, speed=0.3, frame_id="aruco_map")
            node.navigate_wait(x=LINE_START[0], y=LINE_START[1], z=node.LINE_HEIGHT, speed=LINE_VEL, frame_id="aruco_map")
            rospy.sleep(5)

            node.first_reverse = False
            node.is_line_enabled = True
            node.objects_searcher.enable()

            while not node.line_end:
                rospy.sleep(1)

            node.is_line_enabled = False
            node.objects_searcher.disable()

        print("Go to land area")
        node.navigate_wait(x=0.0, y=0.0, z=node.FLIGHT_HEIGHT, speed=0.3, frame_id="aruco_map")
        node.land()

        rospy.sleep(5)

        rospy.spin()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()