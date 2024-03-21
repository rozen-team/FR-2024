import cv2
import numpy as np
import tf
import tf2_ros
import rospy
import tf2_geometry_msgs
from geometry_msgs.msg import Point32, Vector3Stamped, Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray

from utils import *

from typing import Tuple, Optional

import math

class LineProcessor:
    def __init__(self, 
                 hsv_threshold, 
                 cm: np.ndarray, 
                 dc: np.ndarray, 
                 tf_buffer = None,
                 step_x: int = 1, 
                 threshold: float = 10, 
                 min_segment_length = 20):
        self.hsv_threshold = hsv_threshold
        self.step_x = step_x
        self.threshold = threshold
        self.min_segment_length = min_segment_length

        self.min_area = 200

        self.cm = cm
        self.dc = dc
        self.tf_buffer = tf_buffer

        self.is_enable = False

        self.width_array = []
        self.start = None
        self.segments = []
        self.max_width = 0
        self.all_xs = []

        self.road_pub = rospy.Publisher("/a/road_viz", MarkerArray, queue_size=1)

    def tf_function(self, vec: np.ndarray) -> np.ndarray:
        centers = [vec]
        centers = np.array(centers).astype(np.float64)
        pnt_img_undist = cv2.undistortPoints(centers.reshape(-1, 1, 2), self.cm, self.dc, None, None).reshape(-1, 2).T
        ray_v = np.ones((3, pnt_img_undist.shape[1]))
        ray_v[:2, :] = pnt_img_undist
        ray_v /= np.linalg.norm(ray_v, axis=0)

        if self.tf_buffer is not None:
            try:
                transform = self.tf_buffer.lookup_transform("aruco_map", "main_camera_optical", rospy.Time())
            except tf2_ros.ConnectivityException:
                print("LookupException")
                return None
            except tf2_ros.LookupException:
                print("LookupException")
                return None

            t_wb = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])

            ray_v = np.array([unpack_vec(tf2_geometry_msgs.do_transform_vector3(Vector3Stamped(vector=Vector3(v[0], v[1], v[2])), transform)) for v in ray_v.T])
            ray_o = t_wb

            pnts = [intersect_ray_plane(v, ray_o) for v in ray_v]
            if pnts[0] is not None:
                return pnts[0][:2]

        return None

    def publish_road(self):
        segments = self.calculate_segments()
        if segments is None:
            return

        result = []

        prev_end = None
        for idx, (start, end, _, width) in enumerate(segments):
            marker = Marker()
            marker.header.frame_id = "aruco_map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "color_markers"
            marker.id = idx
            marker.type =  Marker.CUBE
            marker.action = Marker.ADD

            if prev_end is not None:
                start = prev_end
            prev_end = end

            w = (width * 0.05) + 0.02

            center_pos = [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0]
            length = math.sqrt(pow(start[0] - end[0], 2) + pow(start[1] - end[1], 2))
            quat = tf.transformations.quaternion_from_euler(0, 0, math.atan2(start[1] - end[1], start[0] - end[0]))

            # Позиция и ориентация
            marker.pose.position.x = center_pos[0]
            marker.pose.position.y = center_pos[1]
            marker.pose.position.z = 0.04
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            
            # Масштаб
            marker.scale.x = length
            marker.scale.y = w
            marker.scale.z = 0.08

            # Цвет
            marker.color.a = 1.0

            marker.color.r = 0.7
            marker.color.g = 0.5
            marker.color.b = 0.0

            result.append(marker)

        # Публикуем маркеры
        self.road_pub.publish(MarkerArray(markers=result))

    def flow(self, image: np.ndarray, hsv: np.ndarray):
        self.publish_road()

        if not self.is_enable:
            return

        self.realtime_line_segmentation(tf_function=self.tf_function, image=image, hsv=hsv)

    def enable(self):
        self.is_enable = True

    def disable(self):
        self.is_enable = False

    def realtime_line_segmentation(self, tf_function, image: np.ndarray, hsv: np.ndarray):
        '''Определение сегментов дороги'''
        bin = cv2.inRange(hsv, self.hsv_threshold[0], self.hsv_threshold[1])

        # нахождение контуров
        cnts, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) > self.min_area]

        if len(cnts) == 0:
            return None

        cnt = max(cnts, key=cv2.contourArea)

        # нахождение минимального описанного прямоугольника
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # классификация отрезков прямоугольника
        if math.dist(box[0], box[1]) > math.dist(box[1], box[2]):
            walls = ((box[0], box[1]), (box[3], box[2]))
            between = ((box[1], box[2]), (box[3], box[0]))
        else:
            walls = ((box[1], box[2]), (box[0], box[3]))
            between = ((box[2], box[3]), (box[0], box[1]))

        corner = walls[0][0]

        wall_height = math.dist(*walls[0])
        wall_width = math.dist(*between[0])

        # нахождение векторов для матрицы перехода
        vec_y = np.array([
            walls[0][1][0] - walls[0][0][0],
            walls[0][1][1] - walls[0][0][1]
        ]) / wall_height

        vec_x = np.array([
            between[0][1][0] - between[0][0][0],
            between[0][1][1] - between[0][0][1]
        ]) / wall_width

        origin = np.array([
            [1, 0],
            [0, 1]
        ])

        dest = np.array([
            vec_x,
            vec_y
        ])

        l = np.vstack([origin[0].T, origin[1].T])
        r = np.vstack([dest[0].T, dest[1].T])

        # нахождение матрицы перехода
        T = np.linalg.solve(l, r)
        T_inv = np.linalg.inv(T)

        space_x = np.linspace(0, int(wall_width), int(wall_width) // self.step_x)

        c = 0
        xs = []
        for x in space_x:
            # Переход от локальных координат дороги к глобальным координатам кадра
            local_vec = np.array([x, 0]).T
            global_vec = T_inv @ local_vec

            coordinates = global_vec.T + corner

            # Подсчёт пикселей дороги
            point = bin[np.clip(int(coordinates[1]), 0, hsv.shape[0]), np.clip(int(coordinates[0]), 0, hsv.shape[1])]
            image[np.clip(int(coordinates[1]), 0, hsv.shape[0]), np.clip(int(coordinates[0]), 0, hsv.shape[1])] = (0, 255, 100)
            if point > 127:
                c += 1
                xs.append(x)

        self.all_xs.append(np.array(xs).mean() if len(xs) > 0 else 0)

        if self.start is None:
            start_vec = np.array([0, 0])
            frame_start_vec = (T_inv @ start_vec.T).T + corner
            map_start_vec = tf_function(frame_start_vec)

            self.start = map_start_vec

        elif abs(c - self.width_array[-1]) > self.threshold:
            end_vec = np.array([0, 0])

            frame_end_vec = (T_inv @ end_vec.T).T + corner
            map_end_vec = tf_function(frame_end_vec)

            segment = (self.start, map_end_vec, np.array(self.width_array).mean())

            self.segments.append(segment)
            self.width_array = []

        self.width_array.append(c * self.step_x)
        if c > self.max_width:
            self.max_width = c

    def calculate_segments(self):
        if len(self.segments) <= 0:
            return None

        min_width = min(s[2] for s in self.segments) + pow(10, -3)

        return [(*s, round(s[2] / min_width)) for s in self.segments]

    def calc_inverse_transform_matrixes(self, segments):
        invTs = []
        for s in segments:
            dest = np.array([
                np.array([s[1][0] - s[0][0], s[1][1] - s[0][1]]) / math.dist(s[0], s[1]),
                np.array([s[1][1] - s[0][1], s[1][0] - s[0][0]]) / math.dist(s[0], s[1]),
            ])
            origin = np.array([
                [1, 0],
                [0, 1]
            ])
            l = np.vstack([origin[0].T, origin[1].T])
            r = np.vstack([dest[0].T, dest[1].T])

            T = np.linalg.solve(l, r)
            invTs.append(np.linalg.inv(T))

        return invTs

    def is_worker_working(self, x, y, distance, segments, invTs):
        """Аргументы:
            x (int): координата x (карта)
            y (int): координата y (карта)
            distance (int): расстояние, при котором работник считается работающем (карта)
            segments (List[...]): список подотрезков
            invTs (List[T]) список обратных матриц перехода для кадого отрезка

            Возвращает:
            bool, int - 1) работает ли 2) подотрезок, над которым работает
        """

        for s, invT in zip(segments, invTs):
            # переход к базису сегмента
            vec = (np.array([x, y]) - s[1]).T
            local_vec = (invT @ vec).T

            # проверка на работу
            if float(local_vec[0]) >= -distance or float(local_vec[0]) <= s[2] + distance:
                return True
        return False