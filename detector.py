import cv2
import numpy as np

from utils import *

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point32, Vector3Stamped, Vector3, Point
import rospy

from visualization_msgs.msg import Marker, MarkerArray
import requests

class ObjectSearcher:
    # Пороговые значения HSV для определения пожаров и пострадавших соответственно
    lower_thr = (
        (0, 50, 80),
        (170, 50, 80)
    )
    upper_thr = (
        (8, 255, 255),
        (180, 255, 255)
    )

    blue_lower_thr = (105, 50, 80)
    blue_upper_thr = (130, 255, 255)

    # Параметры определения пожаров
    fire_fraction = 0.0035
    fire_radius = 1

    def __init__(self, cm: Optional[np.ndarray] = None, dc: Optional[np.ndarray] = None, tf_buffer = None, cv_bridge = None):
        # Параметры камеры для функции undistort
        self.cm = cm
        self.dc = dc

        # TF буффер и cv_bridge для сообщений типа Image
        self.tf_buffer = tf_buffer
        self.cv_bridge = cv_bridge

        self.objects = []

        self.is_start = False

        self.debug_pub = rospy.Publisher("/a/objects_debug", Image, queue_size=1)
        self.mask_pub = rospy.Publisher("/a/objects_mask", Image, queue_size=1)
        self.objects_pub = rospy.Publisher("/a/objects_viz", MarkerArray, queue_size=1)

    # Метод создает маску по заданным пороговым значениям (для определения пожаров)
    def mask_overlay(self, frame):
        mask = np.zeros(frame.shape[:2], dtype="uint8")

        for lower, upper in zip(self.lower_thr, self.upper_thr):
            mask = cv2.bitwise_or(cv2.inRange(frame, lower, upper), mask)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        mask_blue = cv2.inRange(frame, self.blue_lower_thr, self.blue_upper_thr)
        mask_blue = cv2.erode(mask_blue, None, iterations=2)
        mask_blue = cv2.dilate(mask_blue, None, iterations=2)


        return [mask, mask_blue]

    # Метод, публикующий маркеры пожаров в rviz
    def publish_markers(self):
        result = []
        iddd = 0
        for fs, idx in self.objects:
            # На основе множества распознаваний одного пострадавшего формируем усредненные координаты
            m = np.mean(fs, axis=0)

            marker = Marker()
            marker.header.frame_id = "aruco_map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "color_markers"
            marker.id = iddd
            marker.type =  Marker.CYLINDER
            marker.action = Marker.ADD

            # Позиция и ориентация
            marker.pose.position.x = m[0]
            marker.pose.position.y = m[1]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Масштаб
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.05

            # Цвет
            marker.color.a = 0.8

            color = [1.0, 0.1, 0.0]
            if idx == 1: color = [0, 0.1, 1.0]

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]

            result.append(marker)
            iddd += 1

        # Публикуем маркеры
        self.objects_pub.publish(MarkerArray(markers=result))
        return None

    # 
    def find_closest(self, point, tuple_obj, idxss):
        distances = []
        for fire in tuple_obj:
            m = np.mean(fire[0], axis=0)

            if fire[1] != idxss:
                distances.append(float('inf'))
                continue

            distances.append((m[0] - point[0]) ** 2 + (m[1] - point[1]) ** 2)
        
        min_dist = min(distances)
        return distances.index(min_dist), min_dist

    def insert_object(self, point, idxss):
        if len(self.objects) == 0:
            self.objects.append([[point], idxss])
            return

        idx, distance = self.find_closest(point, self.objects, idxss)
        if distance <= self.fire_radius:
            self.objects[idx][0].append(point)
            return
        self.objects.append([[point], idxss])

    # Вспомогательный метод для определения расстояния между 2 точками
    def distance(self, a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def enable(self):
        self.is_start = True

    def disable(self):
        self.is_start = False

    def on_frame(self, frame, mask_floor, hsv: Optional[np.ndarray] = None):
        self.publish_markers()

        if not self.is_start:
            return

        if hsv is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Создаем маски для нахождения пожаров и пострадавших
        debug = frame.copy()
        mask_overlays = self.mask_overlay(hsv)
        
        """
        # Создаем маску для пола площадки
        contours_floor = cv2.findContours(mask_floor, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE)[-2]

        if len(contours_floor) == 0:
            return

        cnt_floor = sorted(contours_floor , key=cv2.contourArea)[-1]

        mannualy_contour = []

        convex_floor = cv2.convexHull(cnt_floor, returnPoints=False)
        defects = cv2.convexityDefects(cnt_floor, convex_floor)

        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt_floor[s][0])
                end = tuple(cnt_floor[e][0])
                far = tuple(cnt_floor[f][0])

                dst = self.distance(start, end)

                mannualy_contour.append(start)
                if dst >= 40:
                    mannualy_contour.append(far)
                mannualy_contour.append(end)

        mannualy_contour = np.array(mannualy_contour).reshape((-1,1,2)).astype(np.int32)
        if len(mannualy_contour) > 0:
            cv2.drawContours(debug, [mannualy_contour], 0, (255,0,0), 3)

        mask_floor = np.zeros(mask_floor.shape, dtype="uint8")
        if len(mannualy_contour) > 0:
            mask_floor = cv2.fillPoly(mask_floor, pts = [mannualy_contour], color=(255,255,255))
        """

        masks = []
        global_mask = np.zeros(frame.shape[:2], dtype="uint8")
        for mask in mask_overlays:
            masks.append(cv2.bitwise_and(mask, mask))
            global_mask = cv2.bitwise_or(masks[-1], global_mask)

        self.mask_pub.publish(self.cv_bridge.cv2_to_imgmsg(global_mask, "mono8"))

        # Проходимся по маскам для нахождения пожаров и пострадавших
        for idx, m in enumerate(masks):
            contours = cv2.findContours(m, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE)[-2]

            frame_vol = np.prod(frame.shape[0:2])

            # Фильтруем объекты по площади
            assert frame_vol != 0
            contours = list(filter(
                    lambda c: (cv2.contourArea(c) / frame_vol) >= self.fire_fraction and (cv2.contourArea(c) / frame_vol) < 0.2, 
                    contours))

            # Находим центры объектов в кадре
            pnt_img = []
            for cnt in contours:
                M = cv2.moments(cnt)

                if M["m00"] == 0:
                    continue

                pnt_img.append(
                        [int(M["m10"] / (M["m00"])),
                        int(M["m01"] / (M["m00"]))])

                cv2.circle(debug, tuple(pnt_img[-1]), 6, (255, 0, 0), 2)

                color = ((0,255,0) if idx == 0 else (0, 0, 255))
                cv2.drawContours(debug, [cnt], 0, color, 3) 

            # Находим координаты объекта, относительно aruco_map
            if len(pnt_img) > 0:
                pnt_img = np.array(pnt_img).astype(np.float64)
                pnt_img_undist = cv2.undistortPoints(pnt_img.reshape(-1, 1, 2), self.cm, self.dc, None, None).reshape(-1, 2).T
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
                    [self.insert_object(p[:2], idx) for p in pnts if p is not None]

        # Публикуем маркеры rviz и изображения дл отладки
        #self.publish_markers()
        self.debug_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug, "bgr8"))