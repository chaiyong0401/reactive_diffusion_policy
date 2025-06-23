# utils.py
# dynamically get the currently running topics
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
from reactive_diffusion_policy.real_world.device_mapping.device_mapping_server import DeviceToTopic
from loguru import logger
try:
    # from xela_server_ros2.msg._sensor_full import SensorFull  # type: ignore
    from xela_server_ros2.msg import SensStream  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    # SensorFull = None
    SensStream = None

def get_topic_and_type(device_to_topic: DeviceToTopic):
    subs_name_type = []

    for camera_name, info in device_to_topic.realsense.items():
        logger.debug(f'camera info: {info}')
        subs_name_type.append((f'/{camera_name}/color/image_raw', Image))

    for camera_name, info in device_to_topic.usb.items():
        subs_name_type.append((f'/{camera_name}/color/image_raw', Image))
        subs_name_type.append((f'/{camera_name}/marker_offset/information', PointCloud2))
        subs_name_type.append((f'/{camera_name}/digit_offset/information', PointCloud2))
    
    if SensStream is not None:
        subs_name_type.append(('/xServTopic', SensStream))
    else:
        logger.warning('xServTopic message type not found; tactile data disabled')
   
    subs_name_type.extend([
        ('/left_tcp_pose', PoseStamped),
        ('/right_tcp_pose', PoseStamped),
        ('/left_gripper_state', JointState),
        ('/right_gripper_state', JointState),
        ('/left_tcp_vel', TwistStamped),
        ('/right_tcp_vel', TwistStamped),
        # ('/left_tcp_wrench', WrenchStamped),
        # ('/right_tcp_wrench', WrenchStamped)
        # ('/left_tcp_taxels', Float32MultiArray),
        # ('/right_tcp_taxels', Float32MultiArray)
    ])

    return subs_name_type


