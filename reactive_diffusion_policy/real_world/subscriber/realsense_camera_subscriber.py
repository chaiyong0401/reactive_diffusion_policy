import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
import cv2
import numpy as np
import open3d as o3d
import time
from loguru import logger

class RealsenseCameraSubscriber(Node):
    def __init__(self, 
                 camera_serial_number: str = '036422060422',
                 camera_name: str = 'camera_base',
                 debug_depth = False,
                 debug_rgb = False
                 ):
        node_name = f'{camera_name}_subscriber'
        super().__init__(node_name)
        self.depth_subscription = self.create_subscription(
            PointCloud2,
            f'/{camera_name}/depth/points',  
            self.depth_listener_callback,
            10
        )
        self.rgb_subscription = self.create_subscription(
            Image,
            f'/{camera_name}/color/image_raw',
            self.rgb_listener_callback,
            10
        )
        self.camera_name = camera_name
        self.prev_time = time.time()
        self.frame_count = 0
        self.debug_depth = debug_depth
        self.debug_rgb = debug_rgb

    def rgb_listener_callback(self, msg):
        try:
            # Decode the image from JPEG format
            np_arr = np.frombuffer(msg.data, np.uint8)
            color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if self.debug_rgb:
                if color_image is not None:
                    # Display the image using OpenCV
                    cv2.imshow("Received Image", color_image)
                    cv2.waitKey(1)
                else:
                    logger.debug('Failed to decode image')

        except Exception as e:
            logger.error(f"Error during image decoding: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Realsense Camera Subscriber')
    parser.add_argument('--camera_name', type=str, default='external_camera',
                        help='Name of the realsense camera')
    args = parser.parse_args()
        
    data_subscriber = RealsenseCameraSubscriber(camera_name = args.camera_name, debug_depth = False)
    rclpy.spin(data_subscriber)

    data_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()