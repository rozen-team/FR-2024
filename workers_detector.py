import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point32, Vector3Stamped, Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray

import cv2
import numpy as np

from utils import *


class WorkersData:
    def __init__(self, shape, thresholds, color):
        self.shape = shape
        self.thresholds = thresholds
        self.color = color

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype="uint8")
        for threshold in self.thresholds:
            lower, upper = threshold
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower, upper), mask)

        return mask

    def marker_color(self):
        return [float(c) / 255.0 for c in self.color]

class WorkersSearcher:
    def __init__(self,
                 workers_data,
                 cm: Optional[np.ndarray] = None, 
                 dc: Optional[np.ndarray] = None, 
                 tf_buffer = None, 
                 bridge = None, 
                 debug_publisher = True):
        # Параметры камеры для функции undistort
        self.cm = cm
        self.dc = dc

        # TF буффер и cv_bridge для сообщений типа Image
        self.tf_buffer = tf_buffer
        self.bridge = bridge

        self.workers_data = workers_data
        self.workers_fraction = 0.0045
        self.workers_radius = 0.5
        self.workers = []

        self.is_enable = False
        self.debug_publisher = debug_publisher

        self.morph_iterations = 2

        if self.debug_publisher:
            self.debug_pub = rospy.Publisher("/a/workers_debug", Image, queue_size=1)
            self.mask_pub = rospy.Publisher("/a/workers_mask", Image, queue_size=1)
        self.workers_pub = rospy.Publisher("/a/workers_viz", MarkerArray, queue_size=1)

    def get_workers(self):
        actual_workers = []
        for points, _ in self.workers:
            m = np.mean(points, axis=0)
            actual_workers.append(m)

        return actual_workers

    # Метод, публикующий маркеры пожаров в rviz
    def publish_markers(self):
        result = []
        for idx, (points, data) in enumerate(self.workers):
            # На основе множества распознаваний одного пострадавшего формируем усредненные координаты
            m = np.mean(points, axis=0)

            marker = Marker()
            marker.header.frame_id = "aruco_map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "color_markers"
            marker.id = idx
            marker.type =  data.shape
            marker.action = Marker.ADD

            # Позиция и ориентация
            marker.pose.position.x = m[0]
            marker.pose.position.y = m[1]
            marker.pose.position.z = 0.04
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Масштаб
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.08

            # Цвет
            marker.color.a = 1.0

            color = data.marker_color()
            marker.color.r = color[2]
            marker.color.g = color[1]
            marker.color.b = color[0]

            result.append(marker)

        # Публикуем маркеры
        self.workers_pub.publish(MarkerArray(markers=result))

    def find_closest(self, point, worker: WorkersData):
        distances = []
        for w in self.workers:
            m = np.mean(w[0], axis=0)

            if w[1] != worker:
                distances.append(float('inf'))
                continue

            distances.append((m[0] - point[0]) ** 2 + (m[1] - point[1]) ** 2)
        
        min_dist = min(distances)
        return distances.index(min_dist), min_dist

    def insert_worker(self, point, worker: WorkersData):
        if len(self.workers) == 0:
            self.workers.append([[point], worker])
            return

        idx, distance = self.find_closest(point, worker)
        if distance <= self.workers_radius:
            self.workers[idx][0].append(point)
            return
        self.workers.append([[point], worker])

    # Вспомогательный метод для определения расстояния между 2 точками
    def distance(self, a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def enable(self):
        self.is_enable = True

    def disable(self):
        self.is_enable = False

    def update(self, image: np.ndarray, hsv: np.ndarray):
        self.publish_markers()

        debug, united_mask = self.on_frame(image, hsv)
        if self.debug_publisher:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, "bgr8"))
            self.mask_pub.publish(self.bridge.cv2_to_imgmsg(united_mask, "mono8"))

    def on_frame(self, image: np.ndarray, hsv: np.ndarray):
        debug = None
        united_mask = None
        if self.debug_publisher: 
            debug = image.copy()
            united_mask = np.zeros(image.shape[:2], dtype="uint8")

        if not self.is_enable:
            return (debug, united_mask)

        if self.debug_publisher:
            for worker in self.workers_data:
                united_mask = cv2.bitwise_or(worker.mask(hsv), united_mask)

            united_mask = cv2.erode(united_mask, None, iterations=self.morph_iterations)
            united_mask = cv2.dilate(united_mask, None, iterations=self.morph_iterations)


        for worker in self.workers_data:
            m = worker.mask(hsv)
            contours = cv2.findContours(m, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE)[-2]

            frame_vol = np.prod(image.shape[0:2])

            # Фильтруем объекты по площади
            assert frame_vol != 0
            contours = list(filter(
                    lambda c: (cv2.contourArea(c) / frame_vol) >= self.workers_fraction and (cv2.contourArea(c) / frame_vol) < 0.2,
                    contours))

            centers = []
            for cnt in contours:
                M = cv2.moments(cnt)

                if M["m00"] == 0:
                    continue

                centers.append(
                        [int(M["m10"] / (M["m00"])),
                        int(M["m01"] / (M["m00"]))])

                if self.debug_publisher:
                    cv2.circle(debug, tuple(centers[-1]), 6, (0, 127, 127), 2)
                    cv2.drawContours(debug, [cnt], 0, worker.color, 4)

            # Находим координаты объекта, относительно aruco_map
            if len(centers) > 0:
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
                        return (debug, united_mask)
                    except tf2_ros.LookupException:
                        print("LookupException")
                        return (debug, united_mask)

                    t_wb = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])

                    ray_v = np.array([unpack_vec(tf2_geometry_msgs.do_transform_vector3(Vector3Stamped(vector=Vector3(v[0], v[1], v[2])), transform)) for v in ray_v.T])
                    ray_o = t_wb

                    pnts = [intersect_ray_plane(v, ray_o) for v in ray_v]
                    [self.insert_worker(p[:2], worker) for p in pnts if p is not None]

        return (debug, united_mask)