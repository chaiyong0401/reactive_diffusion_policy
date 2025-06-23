'''
This file reads mapping from the api server
and launches all the nodes
'''

import multiprocessing
import threading
import time
import os
import psutil
import signal

import requests
import hydra
from omegaconf import DictConfig, OmegaConf
import rclpy
from loguru import logger

from reactive_diffusion_policy.real_world.publisher.realsense_camera_publisher import RealsenseCameraPublisher
from reactive_diffusion_policy.real_world.publisher.usb_camera_publisher import UsbCameraPublisher
# from reactive_diffusion_policy.real_world.publisher.gelsight_camera_publisher import GelsightCameraPublisher
from reactive_diffusion_policy.real_world.publisher.mctac_camera_publisher import MCTacCameraPublisher
from reactive_diffusion_policy.real_world.publisher.gopro_camera_publisher import UvcCameraPublisher
from reactive_diffusion_policy.real_world.device_mapping.device_mapping_server import DeviceToTopic, DeviceMappingServer

# add this to prevent assigning too may threads when using numpy
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
# add this to prevent assigning too may threads when using open-cv
cv2.setNumThreads(12)

import ctypes
libc = ctypes.CDLL("libc.so.6")
SCHED_RR = 2  # real-time scheduling policy
class SchedParam(ctypes.Structure):
    _fields_ = [("sched_priority", ctypes.c_int)]
param = SchedParam()
param.sched_priority = 99  # highest priority

# pid = os.getpid()
# if libc.sched_setscheduler(pid, SCHED_RR, ctypes.byref(param)) != 0:
#     raise OSError("Failed to set scheduler")

class CameraWorker:
    def __init__(self, camera_config):
        self.camera_config = camera_config
        if camera_config.camera_type == 'D400':
            self.camera_publisher = RealsenseCameraPublisher(**camera_config)
        elif camera_config.camera_type == 'USB':
            self.camera_publisher = UsbCameraPublisher(**camera_config)
        # elif camera_config.camera_type == 'gelsight':
        #     self.camera_publisher = GelsightCameraPublisher(**camera_config)
        elif camera_config.camera_type == 'MCTac':
            self.camera_publisher = MCTacCameraPublisher(**camera_config)
        elif camera_config.camera_type == 'gopro':
            self.camera_publisher = UvcCameraPublisher(**camera_config)
        else:
            raise NotImplementedError
    def handle_signal(self, signum, frame):
        self.camera_publisher.stop()
        logger.info(f"Stopped {self.camera_config.camera_name} camera publisher")
        self.camera_publisher.destroy_node()
        # rclpy.shutdown()

def start_camera_publisher(camera_config):
    # bind the process to the specific cpu core to prevent jitter
    cpu_core_id = set(camera_config.cpu_core_id)
    total_cores = psutil.cpu_count()
    for id in cpu_core_id:
        if id >= total_cores:
            raise ValueError(f"Invalid cpu_id: {id}, total cores: {total_cores}")
    os.sched_setaffinity(0, cpu_core_id)
    camera_config = OmegaConf.to_container(camera_config, resolve=True)
    camera_config.pop("cpu_core_id")
    camera_config = OmegaConf.create(camera_config)

    rclpy.init(args=None)
    worker = CameraWorker(camera_config)
    signal.signal(signal.SIGUSR1, worker.handle_signal)

    logger.info(f"Starting {camera_config.camera_name} camera publisher")
    rclpy.spin(worker.camera_publisher)

@hydra.main(
    config_path="reactive_diffusion_policy/config", config_name="real_world_env", version_base="1.3"
)
def main(cfg: DictConfig):
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    try:
        device_mapper_server = DeviceMappingServer(publisher_cfg=cfg.task.publisher,
                                                   **cfg.task.device_mapping_server)
        device_mapping_thread = threading.Thread(target=device_mapper_server.run, daemon=True)
        device_mapping_thread.start()
        time.sleep(1)

        # require the latest mapping of name and topic from fastAPI
        response = requests.get(f"http://{cfg.task.device_mapping_server.host_ip}:{cfg.task.device_mapping_server.port}/get_mapping")
        device_to_topic = DeviceToTopic.model_validate(response.json())
        print("device_to_topic:", device_to_topic)

        # launch the subprocesses based on the mapping from fastapi server
        processes = []
        # Handle realsense cameras
        # for camera_name, camera_info in device_to_topic.realsense.items():
        #     camera_config = None
        #     for cam in cfg.task.publisher.realsense_camera_publisher:
        #         if cam.camera_name == camera_name:
        #             camera_config = cam
        #             break
        #     if camera_config:
        #         # Assigning device_id and type
        #         OmegaConf.set_struct(camera_config, False)
        #         camera_config.camera_serial_number = camera_info.device_id
        #         OmegaConf.set_struct(camera_config, True)

        #         p = multiprocessing.Process(target=start_camera_publisher, args=(camera_config,))
        #         processes.append(p)
        #         p.start()
        
        # handle gopro camera
        for camera_name, camera_info in device_to_topic.gopro.items():
            camera_config = None
            print("camera_name: ", camera_name)
            print("camera_info: ", camera_info)
            for cam in cfg.task.publisher.gopro_camera_publisher:
                if cam.camera_name == camera_name:
                    camera_config = cam
                    break
            if camera_config:
                # Assigning device_id and type
                OmegaConf.set_struct(camera_config, False)
                camera_config.camera_serial_number = camera_info.device_id
                OmegaConf.set_struct(camera_config, True)
                print("camera_serial_number:", camera_config.camera_serial_number)

                p = multiprocessing.Process(target=start_camera_publisher, args=(camera_config,))
                processes.append(p)
                p.start()

        # Handle usb cameras
        for camera_name, camera_info in device_to_topic.usb.items():
            camera_config = None
            for cam in cfg.task.publisher.usb_camera_publisher:
                if cam.camera_name == camera_name:
                    camera_config = cam
                    break
            if camera_config:
                # Assigning device_id and type
                OmegaConf.set_struct(camera_config, False)
                camera_config.camera_index = camera_info.device_id
                OmegaConf.set_struct(camera_config, True)

                p = multiprocessing.Process(target=start_camera_publisher, args=(camera_config,))
                processes.append(p)
                p.start()

        device_mapping_thread.join()
    except KeyboardInterrupt:
        for p in processes:
            os.kill(p.pid, signal.SIGUSR1)
        time.sleep(2)
    finally:
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        logger.info("All Camera publishers shutdown")

if __name__ == "__main__":
    main()