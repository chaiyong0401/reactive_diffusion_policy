import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.time import Time
from cv_bridge import CvBridge

import cv2
import numpy as np
import time
from loguru import logger
import glob
import os

class UvcCameraPublisher(Node):
    def __init__(self, camera_name='gopro_camera', decimate: int = 2, device_id='/dev/video0', width=1280, height=720, fps=30, camera_serial_number=0000, camera_type = 'gopro', rgb_resolution = (224,224)):
        super().__init__(f'{camera_name}_publisher')
        self.device_id = device_id
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_serial_number = camera_serial_number
        self.camera_type = camera_type
        self.rgb_resolution = rgb_resolution
        self.setup_video_file("/home/embodied-ai/mcy/reactive_diffusion_policy_umi/rdp_dataset/tactile_video/GX010621.MP4") ########################
        self.video_path = "/home/embodied-ai/mcy/reactive_diffusion_policy_umi/rdp_dataset/tactile_video/GX010621.MP4"
        node_name = f'{camera_name}_publisher'
        super().__init__(node_name)

        self.image_publisher = self.create_publisher(Image, f"/{camera_name}/color/image_raw", 10)
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)

        self.prev_time = time.time()
        self.frame_count = 0
        self.fps_list = []

        self.bridge = CvBridge()
        self.cap = None
        self.get_logger().info(f"{camera_name} initialized on {device_id}")

        self.start()

    def start(self):

        self.cap = cv2.VideoCapture(self.device_id) ## original ###################################3
        # self.cap = cv2.VideoCapture(self.video_path) # 저장된 video folder ###############################
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open device {self.device_id}")
            raise RuntimeError(f"Cannot open device {self.device_id}")

        self.get_logger().info(f"{self.camera_name} started on {self.device_id}")

    def stop(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info(f"Camera {self.device_id} stopped")
        self.cap = None


    def setup_image_folder(self, folder_path): # 카메라로부터 이미지를 받는게 아니라 폴더로부터 이미지 읽어오는 것 
        self.image_paths = sorted(glob.glob(os.path.join(folder_path, "*.png")))  # 또는 "*.jpg"
        self.image_index = 0

    def setup_video_file(self, video_path): # 카메라로부터 이미지를 받는게 아니라 폴더로부터 동영상 읽어오는 것 
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open video file: {video_path}")
        self.frame_count = 0
        self.prev_time = time.time()    

    def timer_callback(self):
        ## original
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().warn("Camera is not opened. Skipping frame capture.")
            return
    
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture frame")
            return
        ######################################################################
        ## 폴더로부터 읽어오기 #################################
        # if self.cap is None or not self.cap.isOpened():
        #     self.get_logger().warn("Video file is not opened. Skipping frame capture.")
        #     return
        
        # ret, frame = self.cap.read()
        # if not ret:
        #     self.get_logger().info("Reached end of video, restarting...")
        #     self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 처음으로 되돌리기
        #     return
        ######################################################3
        cv2.imwrite("test_frame2.jpg", frame)
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_frame"
        self.image_publisher.publish(msg)

        # calculate fps
        self.frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        if elapsed_time >= 1.0:
            frame_rate = self.frame_count / elapsed_time
            self.fps_list.append(frame_rate)
            logger.debug(f"Frame rate: {frame_rate:.2f} FPS")
            self.prev_time = current_time
            self.frame_count = 0

        # Optional: visualize
        # cv2.imshow("UVC Camera", frame)
        # cv2.waitKey(1)

    def destroy_node(self):
        if self.cap is not None:
            if self.cap.isOpened():
                self.cap.release()
            self.cap = None
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = UvcCameraPublisher(camera_name='gopro_camera', device_id='/dev/video0')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
