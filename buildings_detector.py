from utils import *

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point32, Vector3Stamped, Vector3, Point
import rospy

from visualization_msgs.msg import Marker, MarkerArray
import requests

from server import *
from enum import Enum

class BuildingsData:
    def __init__(self, thresholds, target_height: int = 0):
        self.colors = {
            1: (1.0, 0.8, 0),
            2: (0, 1.0, 0),
            3: (0, 0, 1.0),
            4: (1.0, 0, 0)
        }

        self.thresholds = thresholds

        self.target_height = target_height
        self.real_height = None
        self.color = self.grab_color()

    def grab_color(self):
        assert self.target_height in self.colors
        return self.colors[self.target_height]

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        assert self.thresholds is not None

        mask = np.zeros(hsv.shape[:2], dtype="uint8")
        for threshold in self.thresholds:
            lower, upper = threshold
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower, upper), mask)

        return mask

class BuildingsDetector:
    class States(Enum):
        Searching = 0
        Determination = 1

    def __init__(self, 
                 buildings_data: List[BuildingsData],
                 rest_cient: RESTClient,
                 cm: Optional[np.ndarray] = None, 
                 dc: Optional[np.ndarray] = None, 
                 tf_buffer = None, 
                 cv_bridge = None,
                 debug_publisher = True):
        # Параметры камеры для функции undistort
        self.cm = cm
        self.dc = dc

        # TF буффер и cv_bridge для сообщений типа Image
        self.tf_buffer = tf_buffer
        self.cv_bridge = cv_bridge

        self.buildings_data = buildings_data
        self.rest_cient = rest_cient

        self.morph_iterations = 2

        self.is_enable = False
        self.state = self.States.Searching

        self.finded_buildings = []
        self.determinated_buildings = []
        self.buildings_radius = 2.0

        self.determinated_building_data = None

        self.debug_publisher = debug_publisher
        if self.debug_publisher:
            self.debug_pub = rospy.Publisher("/a/buildings_debug", Image, queue_size=1)
            self.mask_pub = rospy.Publisher("/a/buildings_mask", Image, queue_size=1)
        self.buildings_pub = rospy.Publisher("/a/buildings_viz", MarkerArray, queue_size=1)

    def enable(self):
        self.is_enable = True

    def disable(self):
        self.is_enable = False

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

    def get_point_of_determinated_building(self, point, building):
        idx, _ = self.find_closest(point, self.determinated_buildings, building)

        return np.mean(self.determinated_buildings[idx][0], axis=0), idx

    def insert_object(self, point, building):
        current_buffer = self.finded_buildings
        if self.state == self.States.Determination:
            current_buffer = self.determinated_buildings
            assert self.determinated_building_data is not None

            if building.target_height != self.determinated_building_data.target_height:
                return

        data = BuildingsData(thresholds=None, target_height=building.target_height)

        if len(current_buffer) == 0:
            current_buffer.append([[point], data])
            return

        idx, distance = self.find_closest(point, current_buffer, data)
        if distance <= self.buildings_radius:
            current_buffer[idx][0].append(point)
            return
        current_buffer.append([[point], data])

    def get_determinated_height(self, point):
        distances = []
        for obj in self.determinated_buildings:
            m = np.mean(obj[0], axis=0)
            distances.append((m[0] - point[0]) ** 2 + (m[1] - point[1]) ** 2)
        
        min_dist = min(distances)
        return self.determinated_buildings[distances.index(min_dist)][1].real_height

    def send_request(self):
        print(f"============BUILDINGS REQUEST============")

        buildings = []
        for fs, data in self.determinated_buildings:
            m = np.mean(fs, axis=0)

            x = round(m[0], 5)
            y = round(m[1], 5)
            building = Building([x, y], data.target_height, data.real_height)
            buildings.append(building)

            print(f"Building coords={x};{y} | target_height={data.target_height} | real_height={data.real_height}")

        code, result = self.rest_cient.post_buildings(buildings)
        assert code == 200
        return result.coords

    def publish_markers(self):
        result = []
        for idx, (fs, data) in enumerate(self.determinated_buildings):
            if data.real_height is None:
                continue

            # На основе множества распознаваний одного пострадавшего формируем усредненные координаты
            m = np.mean(fs, axis=0)

            marker = Marker()
            marker.header.frame_id = "aruco_map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "color_markers"
            marker.id = idx
            marker.type =  Marker.CUBE
            marker.action = Marker.ADD

            # Позиция и ориентация
            marker.pose.position.x = m[0]
            marker.pose.position.y = m[1]
            marker.pose.position.z = (data.real_height * 0.25) / 2.0
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Масштаб
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = data.real_height * 0.25

            # Цвет
            marker.color.a = 0.8

            color = data.grab_color()
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]

            result.append(marker)

        # Публикуем маркеры
        self.buildings_pub.publish(MarkerArray(markers=result))

    def update(self, image: np.ndarray, hsv: np.ndarray):
        self.publish_markers()
        debug, mask = self.on_frame(image=image, hsv=hsv)
        if self.debug_publisher:
            self.mask_pub.publish(self.cv_bridge.cv2_to_imgmsg(mask, "mono8"))
            self.debug_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug, "bgr8"))

    def on_frame(self, image: np.ndarray, hsv: np.ndarray):
        debug = None
        united_mask = None
        if self.debug_publisher: 
            debug = image.copy()
            united_mask = np.zeros(image.shape[:2], dtype="uint8")


        if not self.is_enable:
            return debug, united_mask

        if self.debug_publisher:
            for building in self.buildings_data:
                united_mask = cv2.bitwise_or(building.mask(hsv), united_mask)

            united_mask = cv2.erode(united_mask, None, iterations=self.morph_iterations)
            united_mask = cv2.dilate(united_mask, None, iterations=self.morph_iterations)
        
        for building in self.buildings_data:
            m = building.mask(hsv)
            contours = cv2.findContours(
                    m, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
            )[-2]

            frame_vol = np.prod(image.shape[0:2])

            # Фильтруем объекты по площади
            assert frame_vol != 0
            contours = list(filter(
                        lambda c: (cv2.contourArea(c) / frame_vol) >= 0.01 and (cv2.contourArea(c) / frame_vol) < 0.2, 
                        contours
            ))

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

            self.cast_points(points, building)

        return debug, united_mask

    def cast_points(self, points, building):
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
                [self.insert_object(p[:2], building) for i, p in enumerate(pnts) if p is not None]
