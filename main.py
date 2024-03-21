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

from workers_detector import *
from buildings_detector import *
from line_follower import *
from line_processor import *
from server import *
import time

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
    FLIGHT_HEIGHT = 2.0
    LINE_HEIGHT = 0.9

    def __init__(self, source: str = "/main_camera/image_raw_throttled", type: VideoSource.SourceType = VideoSource.SourceType.Topic):
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.land = rospy.ServiceProxy('land', Trigger)

        self.bridge = CvBridge()
        self.is_enable = False
        self.is_screenshot_handled = False

        self.screenshot_name = 'empty'

        self.video_source = VideoSource(callback=self.callback, source=source, type=type, bridge=self.bridge)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.workers_searcher = WorkersSearcher(
            workers_data=
                [
                    WorkersData(shape=Marker.CUBE, thresholds=
                        [
                            ((105, 50, 80), (130, 255, 255))
                        ], 
                        color=(150, 70, 0)
                    ),
                    WorkersData(shape=Marker.CYLINDER, thresholds=
                        [
                            ((0, 50, 80), (8, 255, 255)),
                            ((170, 50, 80), (180, 255, 255))
                        ], 
                        color=(0, 70, 150)
                    )
                ],
            cm=self.video_source.cm, 
            dc=self.video_source.dc, 
            tf_buffer=self.tf_buffer, 
            bridge=self.bridge, 
            debug_publisher=True
        )

        self.buildings_searcher = BuildingsDetector(cm=self.video_source.cm, dc=self.video_source.dc, tf_buffer=self.tf_buffer, cv_bridge=self.bridge)

        self.line_follower = LineFollower(
            line_threshold=[(23, 48, 141), (52, 180, 255)],
            bridge=self.bridge,
            target_height=self.LINE_HEIGHT, 
            debug_publisher=True, 
            k_velocity_z=0.35
        )

        self.line_processor = LineProcessor(
            hsv_threshold=[(23, 48, 141), (52, 180, 255)],
            cm=self.video_source.cm, 
            dc=self.video_source.dc, 
            tf_buffer=self.tf_buffer
        )

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
        self.is_enable = True

    def disable(self):
        self.is_enable = False

    # Callback-метод топика с изображением
    def callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        if not self.is_enable:
            return

        if self.is_screenshot_handled:
            cv2.imwrite(f'{self.screenshot_name}.png', image)
            self.is_screenshot_handled = False

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        self.line_follower.update(image=image, hsv=hsv)
        self.line_processor.flow(image=image.copy(), hsv=hsv)
        self.workers_searcher.update(image=image, hsv=hsv)

        self.buildings_searcher.on_frame(image)

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

    LINE_START = (0.5, 0.5)
    #LINE_HEIGHT = 1.0
    LINE_VEL = 0.2

    ENABLE_BUILDINGS = False
    ENABLE_LINE = True


    try:
        #node = NodeHandle(source="output_1.avi", type=VideoSource.SourceType.File)
        node = NodeHandle()
        rest = RESTClient("dfd3bad5-4ff7-4826-a2be-ee3b7bde078b")

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
            node.navigate_wait(x=LINE_START[0], y=LINE_START[1], z=node.line_follower.target_height, speed=LINE_VEL, frame_id="aruco_map")
            rospy.sleep(5)

            node.line_follower.enable()
            node.workers_searcher.enable()
            node.line_processor.enable()

            while not (node.line_follower.state == LineFollower.States.End):
                rospy.sleep(1)

            node.line_follower.disable()
            node.workers_searcher.disable()
            node.line_processor.disable()

            print("========Road information========")
            segments = node.line_processor.calculate_segments()
            if segments is None:
                segments = []

            segments_ser = []
            prev_end = None
            road_length = 0
            for (start, end, _, width) in segments:
                if prev_end is not None:
                    start = prev_end

                length = math.sqrt(pow(start[0] - end[0], 2) + pow(start[1] - end[1], 2))
                segment = Segment(
                    border=Border(
                        start=(start[0], start[1]),
                        end=(end[0], end[1])
                    ),
                    length=length
                )
                road_length += length
                prev_end = end

                print(f"start=({start[0]}, {start[1]}); end=({end[0]}, {end[1]}); length={length}")

                segments_ser.append(segment)
            print(f"Road length={road_length}")

            print("========Workers information========")
            workers = node.workers_searcher.get_workers()
            workers_ser = []
            for (x, y) in workers:
                invTs = node.line_processor.calc_inverse_transform_matrixes(segments)
                is_working = node.line_processor.is_worker_working(x, y, 0.3, segments, invTs)

                worker = Worker(
                            coords=(x, y),
                            is_working=is_working
                        )
                print(f"coords=({round(x, 3)}, {round(y, 3)}); is_working={is_working}")
                workers_ser.append(worker)

            code = rest.post_roads([
                Road(
                    length=road_length,
                    segments=segments_ser,
                    workers=workers_ser
                )
            ])
            print(f'Send post message /roads to server...{code}')

        print("Go to land area")
        node.navigate_wait(x=0.0, y=0.0, z=node.FLIGHT_HEIGHT, speed=0.3, frame_id="aruco_map")
        node.land()

        rospy.sleep(5)

        rospy.spin()

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()