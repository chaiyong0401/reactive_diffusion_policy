from sensor_msgs.msg import PointCloud2, JointState, Image
from geometry_msgs.msg import PoseStamped, VelocityStamped, WrenchStamped, TwistStamped
import numpy as np
import open3d as o3d
import cv2
import copy
from typing import Dict, Tuple, List, Optional
from loguru import logger
from cv_bridge import CvBridge
from reactive_diffusion_policy.common.space_utils import ros_pose_to_6d_pose
from reactive_diffusion_policy.common.data_models import SensorMessage
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.common.visualization_utils import visualize_pcd_from_numpy, visualize_rgb_image
from std_msgs.msg import Float32MultiArray
try:
    from xela_server_ros2.msg import SensStream  # type: ignore
    # from xela_server_ros2.msg import SensorFull  # type: ignore
    logger.debug("xela_server_ros2 package found, using SensorFull message")
except Exception:  # pragma: no cover - optional dependency
    SensStream = None

class ROS2DataConverter:
    """
    Data converter class that converts ROS2 topic data into Pydantic data models
    """
    def __init__(self,
                 transforms: RealWorldTransforms,
                 depth_camera_point_cloud_topic_names: List[Optional[str]] = [None, None, None],  # external, left wrist, right wrist
                 depth_camera_rgb_topic_names: List[Optional[str]] = [None, None, None],  # external, left wrist, right wrist
                 tactile_camera_rgb_topic_names: List[Optional[str]] = [None, None, None, None],  # left gripper1, left gripper2, right gripper1, right gripper2
                 tactile_camera_marker_topic_names: List[Optional[str]] = [None, None, None, None], # left gripper1, left gripper2, right gripper1, right gripper2
                 tactile_camera_marker_dimension: int = 2,
                 tactile_camera_digit_topic_name: List[Optional[str]] = [None],
                 debug = True):
        self.transforms = transforms
        self.debug = debug
        self.depth_camera_point_cloud_topic_names = depth_camera_point_cloud_topic_names
        self.depth_camera_rgb_topic_names = depth_camera_rgb_topic_names
        self.tactile_camera_rgb_topic_names = tactile_camera_rgb_topic_names
        self.tactile_camera_marker_topic_names = tactile_camera_marker_topic_names
        self.tactile_camera_digit_topic_name = tactile_camera_digit_topic_name
        self.bridge = CvBridge()
        self.tactile_camera_marker_dimension = tactile_camera_marker_dimension
        self.xela_taxel_mean = np.load('/home/embodied-ai/mcy/reactive_diffusion_policy_umi/data/tactile_normalization_mean.npy')
        self.xela_taxel_std = np.load('/home/embodied-ai/mcy/reactive_diffusion_policy_umi/data/tactile_normalization_std.npy')

    def visualize_tcp_poses(self, tcp_pose_left_in_world: np.ndarray, tcp_pose_right_in_world: np.ndarray):
        world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        left_tcp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        left_tcp.transform(tcp_pose_left_in_world)

        left_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        left_base.transform(self.transforms.left_robot_base_to_world_transform)

        right_tcp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        right_tcp.transform(tcp_pose_right_in_world)

        right_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        right_base.transform(self.transforms.right_robot_base_to_world_transform)

        o3d.visualization.draw_geometries([world, left_tcp, left_base, right_tcp, right_base])

    # ROS에서 publish한 PoseStamped, TwistStamped, WrenchStamped, JointState를 numpy array로 전환 
    def convert_robot_states(self, topic_dict: Dict) -> (
            Tuple)[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left_tcp_pose: PoseStamped = topic_dict['/left_tcp_pose']
        right_tcp_pose: PoseStamped = topic_dict['/right_tcp_pose']

        left_gripper_state: JointState = topic_dict['/left_gripper_state']
        right_gripper_state: JointState = topic_dict['/right_gripper_state']

        left_tcp_vel: TwistStamped = topic_dict['/left_tcp_vel']
        right_tcp_vel: TwistStamped = topic_dict['/right_tcp_vel']

        # left_tcp_wrench: WrenchStamped = topic_dict['/left_tcp_wrench']
        # right_tcp_wrench: WrenchStamped = topic_dict['/right_tcp_wrench']
         # tactile data published as Float32MultiArray from 16 taxels
        # tactile data from dedicated sensor topic
        left_tcp_wrench_array = np.zeros(16) 
        right_tcp_wrench_array = np.zeros(16) 

        if SensStream is not None and '/xServTopic' in topic_dict:
            # logger.debug("xServTopic found")
            left_tcp_wrench_array, right_tcp_wrench_array = self.decode_xserv_topic(
                topic_dict['/xServTopic'])
        else:
            logger.debug("xServTopic not found, using Float32MultiArray")
            # left_tcp_wrench: Float32MultiArray = topic_dict.get('/left_tcp_taxels')
            # right_tcp_wrench: Float32MultiArray = topic_dict.get('/right_tcp_taxels')
            # logger.debug(f"left_tcp_wrench: {left_tcp_wrench}, right_tcp_wrench: {right_tcp_wrench}")
            # left_tcp_wrench_array = np.array(left_tcp_wrench.data) if left_tcp_wrench is not None else np.zeros(16, dtype=np.float32)
            # right_tcp_wrench_array = np.array(right_tcp_wrench.data) if right_tcp_wrench is not None else np.zeros(16, dtype=np.float32)


        left_tcp_pose_array = ros_pose_to_6d_pose(left_tcp_pose.pose)
        # logger.info(f"left_tcp_pose_in_convert_robot_states: {left_tcp_pose_array}")
        right_tcp_pose_array = ros_pose_to_6d_pose(right_tcp_pose.pose)

        left_tcp_vel_array = np.array([left_tcp_vel.twist.linear.x, left_tcp_vel.twist.linear.y, left_tcp_vel.twist.linear.z,
                                 left_tcp_vel.twist.angular.x, left_tcp_vel.twist.angular.y,
                                 left_tcp_vel.twist.angular.z])
        right_tcp_vel_array = np.array(
            [right_tcp_vel.twist.linear.x, right_tcp_vel.twist.linear.y, right_tcp_vel.twist.linear.z,
             right_tcp_vel.twist.angular.x, right_tcp_vel.twist.angular.y, right_tcp_vel.twist.angular.z])

        # left_tcp_wrench_array = np.array(
        #     [left_tcp_wrench.wrench.force.x, left_tcp_wrench.wrench.force.y, left_tcp_wrench.wrench.force.z,
        #      left_tcp_wrench.wrench.torque.x, left_tcp_wrench.wrench.torque.y, left_tcp_wrench.wrench.torque.z])
        # right_tcp_wrench_array = np.array(
        #     [right_tcp_wrench.wrench.force.x, right_tcp_wrench.wrench.force.y, right_tcp_wrench.wrench.force.z,
        #      right_tcp_wrench.wrench.torque.x, right_tcp_wrench.wrench.torque.y, right_tcp_wrench.wrench.torque.z])
        # left_tcp_wrench_array = np.array(left_tcp_wrench.data)
        # right_tcp_wrench_array = np.array(right_tcp_wrench.data)


        left_gripper_state_array = np.array([left_gripper_state.position[0], left_gripper_state.effort[0]])
        right_gripper_state_array = np.array([right_gripper_state.position[0], right_gripper_state.effort[0]])

        return (left_tcp_pose_array, right_tcp_pose_array, left_tcp_vel_array, right_tcp_vel_array,
                left_tcp_wrench_array, right_tcp_wrench_array, left_gripper_state_array, right_gripper_state_array)
    
    def decode_depth_rgb_image(self, msg: Image) -> np.ndarray:
        # Decode the image from JPEG format
        np_arr = np.frombuffer(msg.data, np.uint8)
        color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        return color_image

    # JPEG RGB이미지를 numpy array로 변환
    def decode_rgb_image(self, msg: Image) -> np.ndarray:
        np_arr = np.frombuffer(msg.data, np.uint8)
        color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        return rgb_image

    # Point array에서 마커 위치 및 offset 추출
    def decode_tactile_messages(self, msg: PointCloud2):
        # TODO: use 3D representation as default
        if self.tactile_camera_marker_dimension == 2:
            data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
            # Decode points array into marker and offsets
            marker_locations = copy.deepcopy(data[:, :2])
            marker_offsets = copy.deepcopy(data[:, 2:4])
        elif self.tactile_camera_marker_dimension == 3:
            data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 6)
            # Decode points array into marker and offsets
            marker_locations = copy.deepcopy(data[:, :3])
            marker_offsets = copy.deepcopy(data[:, 3:6])
        else:
            raise ValueError(f"Invalid tactile camera marker dimension: {self.tactile_camera_marker_dimension}")

        # We don't need to un-normalize the marker locations and offsets
        return marker_locations, marker_offsets

    def decode_xserv_topic(self, msg: SensStream):
        """Decode tactile forces from custom Xela sensor message."""
        left_taxels = np.zeros(16, dtype=np.float32)
        right_taxels = np.zeros(16, dtype=np.float32)
        try:
            mean = self.xela_taxel_mean  # shape (16,)
            std = self.xela_taxel_std    # shape (16,)

            sensors = getattr(msg, 'sensors', [])
            for sensor in sensors:
                taxels = getattr(sensor, 'taxels', [])
                values = [float(getattr(f, 'z', 0.0)) for f in taxels]
                values = values[:16]
                pad = 16 - len(values)
                if pad > 0:
                    values.extend([0.0] * pad)

                normed_values = (np.array(values, dtype=np.float32) - mean) / (std + 1e-8)

                # if getattr(sensor, 'sensor_pos', 0) == 0:
                #     left_taxels = np.array(values, dtype=np.float32)
                #     logger.debug(f'Left taxels: {left_taxels}')
                # else:
                #     right_taxels = np.array(values, dtype=np.float32)
                #     logger.debug(f'Right taxels: {right_taxels}')
                left_taxels = normed_values
                right_taxels = normed_values
        except Exception as e:
            logger.warning(f'Failed to decode xServTopic: {e}')
        return left_taxels, right_taxels
    
    def convert_depth_camera(self, topic_dict: Dict) -> \
            Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        point_cloud_list = []
        for idx, topic_name in enumerate(self.depth_camera_point_cloud_topic_names):
            # Not supported yet
            point_cloud_list.append(None)

        rgb_image_list = []
        for idx, topic_name in enumerate(self.depth_camera_rgb_topic_names):
            if topic_name is not None:
                if self.debug:
                    logger.debug(topic_name)
                assert topic_name in topic_dict, f"Topic {topic_name} not found in topic_dict"
                rgb_image_list.append(self.decode_depth_rgb_image(topic_dict[topic_name]))
            else:
                rgb_image_list.append(None)

        return point_cloud_list, rgb_image_list

    # Tactile camera의 RGB 및 tactile marker 추출
    def convert_tactile_camera(self, topic_dict: Dict) -> \
            Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]],List[Optional[np.ndarray]]]:
        rgb_image_list = []
        for idx, topic_name in enumerate(self.tactile_camera_rgb_topic_names):
            if topic_name is not None:
                if self.debug:
                    logger.debug(topic_name)
                assert topic_name in topic_dict, f"Topic {topic_name} not found in topic_dict"
                rgb_image_list.append(self.decode_rgb_image(topic_dict[topic_name]))
            else:
                rgb_image_list.append(None)

        marker_loc_list = []
        for idx, topic_name in enumerate(self.tactile_camera_marker_topic_names):
            if topic_name is not None:
                if self.debug:
                    logger.debug(topic_name)
                logger.debug(f"tactile_camera_marker_topic_names: {topic_name}")
                assert topic_name in topic_dict, f"Topic {topic_name} not found in topic_dict"
                marker_loc_list.append((self.decode_tactile_messages(topic_dict[topic_name]))[0])
            else:
                # logger.debug("None")
                marker_loc_list.append(None)

        marker_offset_list= []
        for idx, topic_name in enumerate(self.tactile_camera_marker_topic_names):
            if topic_name is not None:
                if self.debug:
                    logger.debug(topic_name)
                logger.debug(f"tactile_camera_marker_topic_names: {topic_name}")
                assert topic_name in topic_dict, f"Topic {topic_name} not found in topic_dict"
                marker_offset_list.append((self.decode_tactile_messages(topic_dict[topic_name]))[1])
            else:
                # logger.debug("None")
                marker_offset_list.append(None)
        # if self.debug:
        #     logger.debug(f'marker_loc_offset_list {np.max(marker_offset_list[0])}, {np.max(marker_offset_list[1])}')

        digit_offset_list= []
        for idx, topic_name in enumerate(self.tactile_camera_digit_topic_name):
            if topic_name is not None:
                if self.debug:
                    logger.debug(topic_name)
                logger.debug(f"tactile_camera_digit_topic_names: {topic_name}")
                assert topic_name in topic_dict, f"Topic {topic_name} not found in topic_dict"
                digit_offset_list.append(((topic_dict[topic_name]))[0])
            else:
                # logger.debug("None")
                digit_offset_list.append(None)
        # if self.debug:
        #     logger.debug(f'digit_loc_offset_list {np.max(marker_offset_list[0])}, {np.max(marker_offset_list[1])}')
        

        return rgb_image_list, marker_loc_list, marker_offset_list, digit_offset_list

    # 모든 데이터를 하나의 SensorMesssage 객체로 통합 (convert tactile, robot_state, depth 합치기)
    def convert_all_data(self, topic_dict: Dict) -> SensorMessage:
        # calculate the lastest timestamp in the topic_dict
        # latest_timestamp = max([msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        #                         for msg in topic_dict.values()])
        # 추가 
        timestamps = []
        for msg in topic_dict.values():
            if hasattr(msg, 'header'):
                timestamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            else:
                # header가 없는 메시지 무시하거나, 적절한 timestamp를 지정
                pass
        latest_timestamp = max(timestamps)

        (left_tcp_pose, right_tcp_pose, left_tcp_vel, right_tcp_vel,
         left_tcp_wrench, right_tcp_wrench, left_gripper_state, right_gripper_state) = (
            self.convert_robot_states(topic_dict))
        depth_camera_pointcloud_list, depth_camera_rgb_list = self.convert_depth_camera(topic_dict)
        tactile_camera_rgb_list, tactile_camera_marker_loc_list, tactile_camera_marker_offset_list, tactile_camera_digit_offset_list = self.convert_tactile_camera(topic_dict)

        # logger.debug(f"left_tcp_wrench: {left_tcp_wrench}, right_tcp_wrench: {right_tcp_wrench}")
        sensor_msg_args = {
            'timestamp': latest_timestamp,
            'leftRobotTCP': left_tcp_pose,
            'rightRobotTCP': right_tcp_pose,
            'leftRobotTCPVel': left_tcp_vel,
            'rightRobotTCPVel': right_tcp_vel,
            'leftRobotTCPWrench': left_tcp_wrench,
            'rightRobotTCPWrench': right_tcp_wrench,
            'leftRobotGripperState': left_gripper_state,
            'rightRobotGripperState': right_gripper_state,
        }

        if depth_camera_pointcloud_list[0] is not None:
            sensor_msg_args['externalCameraPointCloud'] = depth_camera_pointcloud_list[0]
        if depth_camera_rgb_list[0] is not None:
            sensor_msg_args['externalCameraRGB'] = depth_camera_rgb_list[0]
        if depth_camera_pointcloud_list[1] is not None:
            sensor_msg_args['leftWristCameraPointCloud'] = depth_camera_pointcloud_list[1]
        if depth_camera_rgb_list[1] is not None:
            sensor_msg_args['leftWristCameraRGB'] = depth_camera_rgb_list[1]
        if depth_camera_pointcloud_list[2] is not None:
            sensor_msg_args['rightWristCameraPointCloud'] = depth_camera_pointcloud_list[2]
        if depth_camera_rgb_list[2] is not None:
            sensor_msg_args['rightWristCameraRGB'] = depth_camera_rgb_list[2]
        
        if tactile_camera_rgb_list[0] is not None:
            sensor_msg_args['leftGripperCameraRGB1'] = tactile_camera_rgb_list[0]
        if tactile_camera_rgb_list[1] is not None:
            sensor_msg_args['leftGripperCameraRGB2'] = tactile_camera_rgb_list[1]
        if tactile_camera_rgb_list[2] is not None:
            sensor_msg_args['rightGripperCameraRGB1'] = tactile_camera_rgb_list[2]
        if tactile_camera_rgb_list[3] is not None:
            sensor_msg_args['rightGripperCameraRGB2'] = tactile_camera_rgb_list[3]

        if tactile_camera_marker_loc_list[0] is not None:
            sensor_msg_args['leftGripperCameraMarker1'] = tactile_camera_marker_loc_list[0]
        if tactile_camera_marker_loc_list[1] is not None:
            sensor_msg_args['leftGripperCameraMarker2'] = tactile_camera_marker_loc_list[1]
        if tactile_camera_marker_loc_list[2] is not None:
            sensor_msg_args['rightGripperCameraMarker1'] = tactile_camera_marker_loc_list[2]
        if tactile_camera_marker_loc_list[3] is not None:
            sensor_msg_args['rightGripperCameraMarker2'] = tactile_camera_marker_loc_list[3]

        if tactile_camera_marker_offset_list[0] is not None:
            sensor_msg_args['leftGripperCameraMarkerOffset1'] = tactile_camera_marker_offset_list[0]
        if tactile_camera_marker_offset_list[1] is not None:
            sensor_msg_args['leftGripperCameraMarkerOffset2'] = tactile_camera_marker_offset_list[1]
        if tactile_camera_marker_offset_list[2] is not None:
            sensor_msg_args['rightGripperCameraMarkerOffset1'] = tactile_camera_marker_offset_list[2]
        if tactile_camera_marker_offset_list[3] is not None:
            sensor_msg_args['rightGripperCameraMarkerOffset2'] = tactile_camera_marker_offset_list[3]
        
        if tactile_camera_digit_offset_list[0] is not None:
            sensor_msg_args['leftGripperCameraDigitOffset'] = tactile_camera_digit_offset_list[0]
        # if tactile_camera_digit_offset_list[0] is not None:
        #     sensor_msg_args['leftGripperCameraDigitOffsety'] = tactile_camera_digit_offset_list[1]
        # if tactile_camera_digit_offset_list[0] is not None:
        #     sensor_msg_args['leftGripperCameraDigitOffset'] = tactile_camera_digit_offset_list[2]
            

        sensor_msg = SensorMessage(**sensor_msg_args)
        
        return sensor_msg
        
    
 

