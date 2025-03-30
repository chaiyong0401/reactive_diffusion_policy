import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image,  PointCloud2
from cv_bridge import CvBridge
from loguru import logger
import numpy as np
import cv2

from reactive_diffusion_policy.common.tactile_marker_utils import process_marker_array, marker_track_visualization, display_motion

class UsbCameraSubscriber(Node):
    def __init__(self,
                camera_index = 0,
                width = 352,
                height = 288,
                camera_name = 'usb_camera',
                debug = False):
        node_name = f'{camera_name}_subscriber' 
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.camera_index = camera_index
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.debug = debug

        # subscriber for rgb data
        self.rgb_subscription = self.create_subscription(
            Image,
            f'/{camera_name}/color/image_raw', 
            self.rgb_callback,
            10)

        # subscriber for marker offset data
        self.marker_subscription = self.create_subscription(
            PointCloud2,
            f'/{camera_name}/marker_offset/information',
            self.marker_callback,
            10
        )
        logger.debug(f'Subscription created on {self.camera_name}, waiting for images...')
        logger.debug(f'Subscription created on {self.camera_name}, waiting for marker information...')
        self.rgb_image = None
        self.marker = None
        self.marker_offset = None

    def rgb_callback(self, msg):
        # Decode the image from JPEG format
        np_arr = np.frombuffer(msg.data, np.uint8)
        color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    def marker_callback(self, msg):
        data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        self.marker, self.marker_offset = process_marker_array(data, self.width, self.height)

        if (self.rgb_image is not None) and (self.debug):
            marker_track_visualization(self.rgb_image, self.marker, self.marker_offset)


def main(args=None):
    rclpy.init(args=args)
        
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Usb Camera Publisher')
    parser.add_argument('--camera_name', type=str, default='left_gripper_camera_1',
                        help='name of the USB camera')
    args = parser.parse_args()
    
    data_subscriber = UsbCameraSubscriber(camera_name = args.camera_name, debug = False)
    rclpy.spin(data_subscriber)

    data_subscriber.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()