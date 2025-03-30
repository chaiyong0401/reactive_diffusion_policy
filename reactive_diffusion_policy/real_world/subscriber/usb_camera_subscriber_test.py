import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from loguru import logger
import numpy as np
import cv2
import os
'''
This file is only for test
storing rgb frames for usb cameras for better testing
'''

class UsbCameraSubscriber(Node):
    def __init__(self,
                camera_index = 0,
                width = 400,
                height = 300,
                camera_name = 'usb_camera'):
        node_name = f'{camera_name}_subscriber' 
        super().__init__(node_name)
        self.bridge = CvBridge()
        self.camera_index = camera_index
        self.camera_name = camera_name
        self.width = width
        self.height = height

        # Create directory for saving images if it doesn't exist
        self.save_dir = os.path.join(os.getcwd(), 'data/test/rgb_8_19_17_45')
        os.makedirs(self.save_dir, exist_ok=True)

        # subscriber for rgb data
        self.rgb_subscription = self.create_subscription(
            Image,
            f'/{camera_name}/color/image_raw', 
            self.rgb_callback,
            10)

        logger.debug(f'Subscription created on {self.camera_name}, waiting for images...')
        self.frame_count = 0

    def rgb_callback(self, msg):
        rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        image_filename = os.path.join(self.save_dir, f'frame_{self.frame_count:04d}.jpg')
        cv2.imwrite(image_filename, self.rgb_image)
        # logger.debug(f'Saved {image_filename}')

        cv2.imshow(f'{self.camera_name} - RGB Image', self.rgb_image)
        cv2.waitKey(1)  # Display the image for a short period

        # Increment frame count
        self.frame_count += 1

   
def main(args=None):
    rclpy.init(args=args)
    
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Usb Camera Publisher')
    parser.add_argument('--camera_name', type=str, default='left_gripper_camera_1',
                        help='name of the USB camera')
    args = parser.parse_args()
    

    data_subscriber = UsbCameraSubscriber(camera_name = args.camera_name)
    rclpy.spin(data_subscriber)

    data_subscriber.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()