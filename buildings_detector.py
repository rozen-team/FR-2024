from utils import *

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point32, Vector3Stamped, Vector3, Point
import rospy

from visualization_msgs.msg import Marker, MarkerArray
import requests

from pyzbar import pyzbar
import requests

class BuildingsData:
    def __init__(self, target_height: float = 0.0):
        self.target_height = round(target_height, 2)
        self.color = self.grab_color()

    def grab_color(self):
        if self.target_height >= 1.0:
            return (1.0, 0, 0)
        if self.target_height >= 0.75:
            return (0, 0, 1.0)
        if self.target_height >= 0.5:
            return (0, 1.0, 0)
        if self.target_height >= 0.25:
            return (1.0, 0.8, 0)

        return (0, 0, 0)

class BuildingsDetector:
    def __init__(self, cm: Optional[np.ndarray] = None, dc: Optional[np.ndarray] = None, tf_buffer = None, cv_bridge = None):
        # Параметры камеры для функции undistort
        self.cm = cm
        self.dc = dc

        # TF буффер и cv_bridge для сообщений типа Image
        self.tf_buffer = tf_buffer
        self.cv_bridge = cv_bridge

        self.is_start = False

        self.objects = []
        self.object_radius = 2.0

        self.debug_pub = rospy.Publisher("/a/buildings_debug", Image, queue_size=1)
        self.buildings_pub = rospy.Publisher("/a/buildings_viz", MarkerArray, queue_size=1)

    def enable(self):
        self.is_start = True

    def disable(self):
        self.is_start = False

    def send_request(self):
        # get requests
        #payload = {'x': position[0], 'y': position[1]}
        #material = requests.get('http://65.108.222.51/check_material', params=payload).text

        # parse coords
        return (1.54635, 3.2342)

    def get_height(self, point):
        distances = []
        for obj in self.objects:
            m = np.mean(obj[0], axis=0)

            distances.append((m[0] - point[0]) ** 2 + (m[1] - point[1]) ** 2)

        min_dist = min(distances)
        return self.objects[distances.index(min_dist)][1].target_height

    def find_closest(self, point, tuple_obj, data):
        distances = []
        for obj in tuple_obj:
            m = np.mean(obj[0], axis=0)

            if obj[1].target_height != data.target_height:
                distances.append(float('inf'))
                continue

            distances.append((m[0] - point[0]) ** 2 + (m[1] - point[1]) ** 2)
        
        min_dist = min(distances)
        return distances.index(min_dist), min_dist

    def insert_object(self, point, color):
        self.target_map = {
            'red': 0.5,
            'yellow': 0.75,
            'blue': 1.0,
            'green': 0.25
        }
        data = BuildingsData(target_height=self.target_map[color])

        if len(self.objects) == 0:
            self.objects.append([[point], data])
            return

        idx, distance = self.find_closest(point, self.objects, data)
        if distance <= self.object_radius:
            self.objects[idx][0].append(point)
            return
        self.objects.append([[point], data])

    def publish_markers(self):
        result = []
        iddd = 0
        for fs, data in self.objects:
            # На основе множества распознаваний одного пострадавшего формируем усредненные координаты
            m = np.mean(fs, axis=0)

            marker = Marker()
            marker.header.frame_id = "aruco_map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "color_markers"
            marker.id = iddd
            marker.type =  Marker.CUBE
            marker.action = Marker.ADD

            # Позиция и ориентация
            marker.pose.position.x = m[0]
            marker.pose.position.y = m[1]
            marker.pose.position.z = data.target_height/2.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Масштаб
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = data.target_height

            # Цвет
            marker.color.a = 0.8

            marker.color.r = data.color[0]
            marker.color.g = data.color[1]
            marker.color.b = data.color[2]

            result.append(marker)
            iddd += 1

        # Публикуем маркеры
        self.buildings_pub.publish(MarkerArray(markers=result))
        return None

    def parse_qr(self, data) -> BuildingsData:
        d = BuildingsData(target_height=float(data[0].split(" ")[-1]))
        return d

    def on_frame(self, image):
        """
        debug = image.copy()

        self.publish_markers()
        
        if self.is_start:
            boxes = pyzbar.decode(image)

            # обрабатываем информацию из QR-кодов
            points = []
            data_obj = []
            for box in boxes:
                data = box.data.decode("utf-8").split('\n')

                debug = cv2.rectangle(debug, (box.rect.left, box.rect.top), \
                        (box.rect.width + box.rect.left, box.rect.height + box.rect.top), \
                        (0, 127, 255), 2)

                points.append([int(box.rect.left + box.rect.width/2.0), 
                               int(box.rect.top + box.rect.height/2.0)])
                data_obj.append(self.parse_qr(data))

            self.cast_points(points, data_obj)


        self.debug_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug, "bgr8"))
        """

        debug = image.copy()
        self.publish_markers()

        self.masks_dict = {
            'blue': [(105, 50, 80), (105, 50, 80)],
            'green': [(0, 0, 0), (0, 0, 0)],
            'red': [(0, 0, 0), (0, 0, 0)],
            'yellow': [(0, 0, 0), (0, 0, 0)]
        }

        if self.is_start:
            masks = dict()

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            global_mask = np.zeros(frame.shape[:2], dtype="uint8")
            for key_mask in self.masks_dict:
                low_thr = self.masks_dict[key_mask]
                up_thr = self.masks_dict[key_mask]

                masks[key_mask] = cv2.inRange(hsv, low_thr, up_thr)
                global_mask = cv2.bitwise_or(masks[key_mask], global_mask)

            self.mask_pub.publish(self.cv_bridge.cv2_to_imgmsg(global_mask, "mono8"))
        
            for key in masks:
                m = masks[key]
                contours = cv2.findContours(m, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE)[-2]

                frame_vol = np.prod(frame.shape[0:2])

                # Фильтруем объекты по площади
                assert frame_vol != 0
                contours = list(filter(
                        lambda c: (cv2.contourArea(c) / frame_vol) >= 0.0035 and (cv2.contourArea(c) / frame_vol) < 0.2, 
                        contours))

                # Находим центры объектов в кадре
                points = []
                for cnt in contours:
                    M = cv2.moments(cnt)

                    if M["m00"] == 0:
                        continue

                    points.append(
                            [int(M["m10"] / (M["m00"])),
                            int(M["m01"] / (M["m00"]))])
                    cv2.drawContours(debug, [cnt], 0, (0, 127, 255), 3)

                self.cast_points(points, key)

            self.debug_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug, "bgr8"))

    def cast_points(self, points, color):
        # Находим координаты объекта, относительно aruco_map
        if len(points) > 0:
            points = np.array(points).astype(np.float64)
            pnt_img_undist = cv2.undistortPoints(points.reshape(-1, 1, 2), self.cm, self.dc, None, None).reshape(-1, 2).T
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
                [self.insert_object(p[:2], color) for i, p in enumerate(pnts) if p is not None]