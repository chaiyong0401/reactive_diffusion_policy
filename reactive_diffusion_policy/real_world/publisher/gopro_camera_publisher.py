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
    def __init__(self, camera_name='gopro_camera', decimate: int = 2, device_id='/dev/video0', width=640, height=480, fps=24, camera_serial_number=0000, camera_type = 'gopro', rgb_resolution = (224,224)):
        super().__init__(f'{camera_name}_publisher')
        self.device_id = device_id
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_serial_number = camera_serial_number
        self.camera_type = camera_type
        self.rgb_resolution = rgb_resolution
        # self.setup_video_file("/home/embodied-ai/mcy/reactive_diffusion_policy_umi/rdp_dataset/tactile_video/GX010621.MP4") ########################
        # self.video_path = "/home/embodied-ai/mcy/reactive_diffusion_policy_umi/rdp_dataset/tactile_video/GX010621.MP4"
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
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)

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
            
        # 확인 로그 출력 추가 (선택)
        # actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        # actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # self.get_logger().info(f"Actual resolution: {actual_width} x {actual_height}")

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
        # cv2.imwrite("test_frame2.jpg", frame)
        # msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        # msg.header.stamp = self.get_clock().now().to_msg()
        # msg.header.frame_id = "camera_color_frame"
        # self.image_publisher.publish(msg)


        img = draw_predefined_mask(frame, color=(0,0,0), 
                            mirror=False, gripper=True, finger=False, use_aa=True)
        frame_resized = cv2.resize(frame, self.rgb_resolution)
        cv2.imwrite("test_frame_resized.jpg", frame_resized)
        msg = self.bridge.cv2_to_imgmsg(frame_resized, encoding='bgr8')
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

def pixel_coords_to_canonical(pts, img_shape=(2028, 2704)):
    coords = (np.asarray(pts) - np.array(img_shape[::-1]) * 0.5) / img_shape[0]
    return coords

def canonical_to_pixel_coords(coords, img_shape=(2028, 2704)):
    pts = np.asarray(coords) * img_shape[0] + np.array(img_shape[::-1]) * 0.5
    return pts

def draw_predefined_mask(img, color=(0,0,0), mirror=True, gripper=True, finger=True, use_aa=False):
    all_coords = list()
    if mirror:
        all_coords.extend(get_mirror_canonical_polygon())
    if gripper:
        all_coords.extend(get_gripper_canonical_polygon())
    if finger:
        all_coords.extend(get_finger_canonical_polygon())
    # all_coords.extend(get_sensor_canonical_polygon())
        
    for coords in all_coords:
        pts = canonical_to_pixel_coords(coords, img.shape[:2])
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img,[pts], color=color, lineType=flag)
    return img

def get_mirror_canonical_polygon():
    left_pts = [
        [540, 1700],
        [680, 1450],
        [590, 1070],
        [290, 1130],
        [290, 1770],
        [550, 1770]
    ]
    resolution = [2028, 2704]
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    right_coords = left_coords.copy()
    right_coords[:,0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords

def get_gripper_canonical_polygon():
    left_pts = [
        [1352, 1730],
        [1100, 1700],
        [650, 1500],
        [0, 1350],
        [0, 2028],
        [1352, 2704]
    ]
    resolution = [2028, 2704]
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    right_coords = left_coords.copy()
    right_coords[:,0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords

def get_finger_canonical_polygon(height=0.37, top_width=0.25, bottom_width=1.4):
    # image size
    resolution = [2028, 2704]
    img_h, img_w = resolution

    # calculate coordinates
    top_y = 1. - height
    bottom_y = 1.
    width = img_w / img_h
    middle_x = width / 2.
    top_left_x = middle_x - top_width / 2.
    top_right_x = middle_x + top_width / 2.
    bottom_left_x = middle_x - bottom_width / 2.
    bottom_right_x = middle_x + bottom_width / 2.

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    # create polygon points for opencv API
    points = [[
        [bottom_left_x, bottom_y],
        [top_left_x, top_y],
        [top_right_x, top_y],
        [bottom_right_x, bottom_y]
    ]]
    coords = pixel_coords_to_canonical(points, img_shape=resolution)
    return coords

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
