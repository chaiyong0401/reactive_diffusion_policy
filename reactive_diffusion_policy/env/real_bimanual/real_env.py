
# 07/30 추가
from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
import cv2
import torch
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.rtde_interpolation_controller import RTDEInterpolationController
from umi.real_world.wsg_controller import WSGController
from umi.real_world.dyros_gripper_controller import DYROSController
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.multi_uvc_camera import MultiUvcCamera, VideoRecorder
from umi.real_world.uvc_camera import UvcCamera
from diffusion_policy.common.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator
)
from umi.common.cv_util import (
    draw_predefined_mask, 
    get_mirror_crop_slices
)
from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
from umi.common.usb_util import reset_all_elgato_devices, get_sorted_v4l_paths
from umi.common.pose_util import pose_to_pos_rot
from umi.common.interpolation_util import get_interp1d, PoseInterpolator
from umi.real_world.xela_receive import RosTactileListener
try:
    import rclpy
    from rclpy.node import Node
except Exception:
    rclpy = None
from loguru import logger

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
from dataclasses import dataclass
import threading

# @dataclass
# class Waypoint:
#     t: float
#     pose: np.ndarray  # shape (6,) for robot, shape (1,) or (k,) for gripper
#     kind: str         # "robot" or "gripper"

# class WaypointLogger:
#     def __init__(self, maxlen=10000):
#         self.buf = deque(maxlen=maxlen)

#     def add(self, t, pose, kind):
#         self.buf.append(Waypoint(t=float(t), pose=np.array(pose, dtype=np.float32), kind=kind))

#     def to_arrays(self, kind="robot"):
#         ts, poses = [], []
#         for w in self.buf:
#             if w.kind == kind:
#                 ts.append(w.t); poses.append(w.pose)
#         return np.array(ts), (np.vstack(poses) if len(poses)>0 else np.empty((0,)))

@dataclass
class Waypoint:
    t: float
    pose: np.ndarray  # shape (6,) for robot, shape (1,) or (k,) for gripper
    kind: str         # "robot" or "gripper"
    group: int =0

class WaypointLogger:
    def __init__(self, maxlen=10000):
        self.buf = deque(maxlen=maxlen)

    def add(self, t, pose, kind, group=0):
        self.buf.append(Waypoint(t=float(t), pose=np.array(pose, dtype=np.float32), kind=kind,  group=int(group)))

    def to_arrays(self, kind="robot", return_groups=False):
        ts, poses, groups = [], [], []
        for w in self.buf:
            if w.kind == kind:
                ts.append(w.t); poses.append(w.pose); groups.append(w.group)
        ts = np.array(ts)
        poses = np.vstack(poses) if len(poses) > 0 else np.empty((0,))
        groups = np.array(groups, dtype=int)
        return (ts,poses,groups) if return_groups else (ts,poses)

class UmiEnv:
    def __init__(self, 
            # required params
            output_dir,
            robot_ip,
            gripper_ip,
            gripper_port=1000,
            # env params
            frequency=20,
            robot_type='ur5',
            # obs
            obs_image_resolution=(224,224),
            max_obs_buffer_size=60,
            obs_float32=False,
            camera_reorder=None,
            no_mirror=False,
            fisheye_converter=None,
            mirror_crop=False,
            mirror_swap=False,
            # timing
            align_camera_idx=0,
            # this latency compensates receive_timestamp
            # all in seconds
            camera_obs_latency=0.125,
            robot_obs_latency=0.0001,
            gripper_obs_latency=0.01,
            robot_action_latency=0.1,
            gripper_action_latency=0.1,
            # all in steps (relative to frequency)
            camera_down_sample_steps=1,
            robot_down_sample_steps=1,
            gripper_down_sample_steps=1,
            # all in steps (relative to frequency)
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # action
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            # robot
            tcp_offset=0.21,
            init_joints=False,
            # vis params
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(960, 960),
            # shared memory
            shm_manager=None
            ):
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()

        self.wp_logger = WaypointLogger(maxlen=20000)
        self._wp_group_id = 0
        self._last_sched_wall_time_robot = -1e18
        self._last_sched_wall_time_grip  = -1e18
        self._wp_lock = threading.Lock()

        # Find and reset all Elgato capture cards.
        # Required to workaround a firmware bug.
        reset_all_elgato_devices()

        # Wait for all v4l cameras to be back online
        time.sleep(0.1)
        v4l_paths = get_sorted_v4l_paths()
        if camera_reorder is not None:
            paths = [v4l_paths[i] for i in camera_reorder]
            v4l_paths = paths

        # compute resolution for vis
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(v4l_paths),
            in_wh_ratio=4/3,
            max_resolution=multi_cam_vis_resolution
        )

        # HACK: Separate video setting for each camera
        # Elagto Cam Link 4k records at 4k 30fps
        # Other capture card records at 720p 60fps
        resolution = list()
        capture_fps = list()
        cap_buffer_size = list()
        video_recorder = list()
        transform = list()
        vis_transform = list()
        print("v4l_paths:",v4l_paths)   # v4l_paths: ['/dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB34322RCJG-video-index0']
        for idx, path in enumerate(v4l_paths):
            print(f"Initializing camera {idx} at {path}")
            if 'Cam_Link_4K' in path:
                # print("calm_link_4k")   # X
                res = (3840, 2160)
                fps = 30
                buf = 3
                bit_rate = 6000*1000
                def tf4k(data, input_res=res):
                    img = data['color']
                    f = get_image_transform(
                        input_res=input_res,
                        output_res=obs_image_resolution, 
                        # obs output rgb
                        bgr_to_rgb=True)
                    img = f(img)
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img
                    return data
                transform.append(tf4k)
            else:
                # print("calm_link_4k x res = 1920,1080") # O
                res = (1920, 1080)
                fps = 60
                buf = 1
                bit_rate = 3000*1000
                stack_crop = (idx==0) and mirror_crop
                is_mirror = None
                if mirror_swap:
                    mirror_mask = np.ones((224,224,3),dtype=np.uint8)
                    mirror_mask = draw_predefined_mask(
                        mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
                    is_mirror = (mirror_mask[...,0] == 0)

                def tf(data, input_res=res, stack_crop=stack_crop, is_mirror=is_mirror):
                    # if 'color' not in data or data['color'] is None:
                    #     print("Data does not contain 'color' or is None.")
                    #     return None
                    
                    img = data['color']
                    if fisheye_converter is None:
                        crop_img = None
                        if stack_crop:
                            slices = get_mirror_crop_slices(img.shape[:2], left=False)
                            crop = img[slices]
                            crop_img = cv2.resize(crop, obs_image_resolution)
                            crop_img = crop_img[:,::-1,::-1] # bgr to rgb
                        f = get_image_transform(
                            input_res=input_res,
                            output_res=obs_image_resolution, 
                            # obs output rgb
                            bgr_to_rgb=True)
                        img = np.ascontiguousarray(f(img))
                        if is_mirror is not None:
                            img[is_mirror] = img[:,::-1,:][is_mirror]
                        img = draw_predefined_mask(img, color=(0,0,0), 
                            mirror=no_mirror, gripper=True, finger=False, use_aa=True)
                        if crop_img is not None:
                            img = np.concatenate([img, crop_img], axis=-1)
                    else:
                        img = fisheye_converter.forward(img)
                        img = img[...,::-1]
                    if obs_float32:
                        img = img.astype(np.float32) / 255
                    data['color'] = img # Transformed data['color'].shape: (224, 224, 3), data['color'].dtype: float32
                    return data
                transform.append(tf)

            resolution.append(res)
            capture_fps.append(fps)
            cap_buffer_size.append(buf)
            video_recorder.append(VideoRecorder.create_hevc_nvenc(
                fps=fps,
                input_pix_fmt='bgr24',
                bit_rate=bit_rate
            ))

            def vis_tf(data, input_res=res):
                img = data['color']
                f = get_image_transform(
                    input_res=input_res,
                    output_res=(rw,rh),
                    bgr_to_rgb=False
                )
                img = f(img)
                data['color'] = img
                return data
            vis_transform.append(vis_tf)

        print("v41_ready")
        camera = MultiUvcCamera(
            dev_video_paths=v4l_paths,
            shm_manager=shm_manager,
            resolution=resolution,
            capture_fps=capture_fps, 
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            get_max_k=max_obs_buffer_size,
            receive_latency=camera_obs_latency,
            cap_buffer_size=cap_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            video_recorder=video_recorder,
            verbose=False
        )
        print("camera_ready")
        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                camera=camera,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        cube_diag = np.linalg.norm([1,1,1])
        j_init = np.array([0,-90,-90,-90,90,0]) / 180 * np.pi
        if not init_joints:
            j_init = None

        if robot_type.startswith('ur5'):
            robot = RTDEInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                frequency=500, # UR5 CB3 RTDE
                lookahead_time=0.1,
                gain=300,
                max_pos_speed=max_pos_speed*cube_diag,
                max_rot_speed=max_rot_speed*cube_diag,
                launch_timeout=3,
                tcp_offset_pose=[0,0,tcp_offset,0,0,0],
                payload_mass=None,
                payload_cog=None,
                joints_init=j_init,
                joints_init_speed=1.05,
                soft_real_time=False,
                verbose=False,
                receive_keys=None,
                receive_latency=robot_obs_latency
                )
        elif robot_type.startswith('franka'):
            print("our robot type is franka")
            robot = FrankaInterpolationController(
                shm_manager=shm_manager,
                robot_ip=robot_ip,
                frequency=200,
                Kx_scale=1.0,
                Kxd_scale=np.array([2.0,1.5,2.0,1.0,1.0,1.0]),
                verbose=False,
                receive_latency=robot_obs_latency
            )
        print("robot type check")
        # gripper = WSGController(
        #     shm_manager=shm_manager,
        #     hostname=gripper_ip,
        #     port=gripper_port,
        #     receive_latency=gripper_obs_latency,
        #     use_meters=True
        # )
        gripper = DYROSController(
            shm_manager=shm_manager,
            # hostname=gripper_ip,
            # port=gripper_port,
            receive_latency=gripper_obs_latency,    # 0.01 임의 설정
            use_meters=True
        )
        print("gripper check")

        tactile_listener = None # 07/28
        horizon = 100   # 07/30
        if rclpy is not None:
            tactile_listener = RosTactileListener('xServTopic', horizon)
        else:
            print('ROS not available; tactile input disabled.')


        self.camera = camera
        self.robot = robot
        self.gripper = gripper
        self.multi_cam_vis = multi_cam_vis
        self.frequency = frequency
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.mirror_crop = mirror_crop
        self.tactile_listener = tactile_listener
        # timing
        self.align_camera_idx = align_camera_idx
        self.camera_obs_latency = camera_obs_latency
        self.robot_obs_latency = robot_obs_latency
        self.gripper_obs_latency = gripper_obs_latency
        self.robot_action_latency = robot_action_latency
        self.gripper_action_latency = gripper_action_latency
        self.camera_down_sample_steps = camera_down_sample_steps
        self.robot_down_sample_steps = robot_down_sample_steps
        self.gripper_down_sample_steps = gripper_down_sample_steps
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_camera_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None

        self.start_time = None
        self._tactile_encoder_ready = False
        self.encoder = None
        self.mean = None
        self.std = None
        self._data_to_gnn_batch = None
        self._XELA_USPA44_COORD = None
        self._XELA_TACTILE_ORI_COORD = None
        self._point_per_sensor = None
        self._torch = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.camera.is_ready and self.robot.is_ready and self.gripper.is_ready
    
    def start(self, wait=True):
        self.camera.start(wait=False)
        self.gripper.start(wait=False)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.gripper.stop(wait=False)
        self.camera.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.camera.start_wait()
        self.gripper.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.gripper.stop_wait()
        self.camera.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb): 
        self.stop()

    # ========= async env API ===========
    def _init_tactile_encoder(self):
    
        if self._tactile_encoder_ready:
            return
        # === 당신의 08/07 블록을 여기로 이동 ===
        # 08/07 3dtacdex
        from tacdex3d.pretraining.models.edcoder import PreModel
        from tacdex3d.robomimic.models.utils import data_to_gnn_batch
        # from tacdex3d.robomimic.models.base_nets import MAEGAT
        from tacdex3d.diffusion_policy.real_world.fk.constants import (
            XELA_USPA44_COORD,
            XELA_TACTILE_ORI_COORD,
        )
        from scipy.spatial.transform import Rotation as R
        self._data_to_gnn_batch = data_to_gnn_batch
        self._XELA_USPA44_COORD = XELA_USPA44_COORD
        self._XELA_TACTILE_ORI_COORD = XELA_TACTILE_ORI_COORD
        self._point_per_sensor = len(self._XELA_USPA44_COORD)
        self._torch = torch
        # 08/07 3dtacdex
        num_heads = 4
        num_out_heads = 1
        # num_hidden = 128
        num_hidden = 16
        num_layers = 3
        residual = False
        attn_drop = 0.0
        in_drop = 0.0
        norm = None
        negative_slope = 0.2
        encoder_type = "gat"
        decoder_type = "gat"
        mask_rate = 0.01
        drop_edge_rate = 0.0
        replace_rate = 0.0


        activation = "prelu"
        loss_fn = "mse"
        alpha_l = 3
        concat_hidden = False
        num_features = 6
        num_nodes = 16
        mask_index = 0
        resultant_type = None
        
        full_model = PreModel(
                in_dim=int(num_features),
                num_hidden=int(num_hidden),
                num_layers=num_layers,
                nhead=num_heads,
                nhead_out=num_out_heads,
                activation=activation,
                feat_drop=in_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=residual,
                encoder_type=encoder_type,
                decoder_type=decoder_type,
                mask_rate=mask_rate,
                norm=norm,
                loss_fn=loss_fn,
                drop_edge_rate=drop_edge_rate,
                replace_rate=replace_rate,
                alpha_l=alpha_l,
                concat_hidden=concat_hidden,
                mask_index=mask_index,
                resultant_type=resultant_type,
                num_nodes=16
            )
        full_model.eval()
        self.encoder = full_model.encoder
         # 기존 pretrained weight
        ckpt_path = "/home/embodied-ai/mcy/universal_manipulation_interface/tacdex3d/pretraining/logging_data/pretrain_four_train:tactile_play_data_train_test:tactile_play_data_test_rpr_0.0__mp_100000_wd_0_gat_gat_62/checkpoint_8500.pt"
        # state_dict =torch.load(ckpt_path, map_location='cpu')
        state_dict = self._torch.load(ckpt_path, map_location='cpu')
        self.encoder.nets.load_state_dict(state_dict)
        for param in self.encoder.parameters():
            param.requires_grad = False
        # stats = np.load("/home/embodied-ai/mcy/universal_manipulation_interface/tacdex3d/output/xela_stack_data_force_stats.npy", allow_pickle=True).item()
        stats = np.load("/home/embodied-ai/mcy/universal_manipulation_interface/tacdex3d/output/xela_data_force_stats.npy", allow_pickle=True).item()
        self.mean = stats["mean"]  # shape: (3,)
        self.std = stats["std"]
        self._tactile_encoder_ready = True

    def get_obs(self) -> dict: # dictionary 형태로 return
        """
        Timestamp alignment policy
        'current' time is the last timestamp of align_camera_idx
        All other cameras, find corresponding frame with the nearest timestamp
        All low-dim observations, interpolate with respect to 'current' time
        
        obs_data = {
            'camera0_rgb': np.array([...]),                # 카메라 0의 RGB 이미지 (HxWx3)
            'camera0_rgb_mirror_crop': np.array([...]),    # 좌우 반전된 crop 이미지 (옵션)
            # 'camera1_rgb': np.array([...]),                # 카메라 1의 RGB 이미지
            'robot0_eef_pos': np.array([...]),             # 로봇 EE 위치 (x, y, z) # (N, 3)
            'robot0_eef_rot_axis_angle': np.array([...]),  # 로봇 EE의 회전 (로드리게스 벡터) # (N, 3)
            'robot0_gripper_width': np.array([...]),       # 그리퍼 폭 (m)  # (N, 1)
            'timestamp': np.array([...])                   # 각 관측값의 타임스탬프 # (N,) (camera_obs_timestamp -> 기준 timestamp)

            'tactile_image
        }
        
        """

        "observation dict"
        assert self.is_ready
        self._init_tactile_encoder()

        # get data
        # 60 Hz, camera_calibrated_timestamp
        # 최근 k개의 데이터(frame, timestampe) 수집하여 last_camera_data로 저장
        # camera_obs_horizon = 2, camera_down_sample_steps =1, frequency =20 , math.ceil(올림) 
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (60 / self.frequency))
        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)

        # 125/500 hz, robot_receive_timestamp
        # robot = FrankaInterpolationController
        last_robot_data = self.robot.get_all_state()    # 최대 k개의 target_pose 
        # both have more than n_obs_steps data

        # 30 hz, gripper_receive_timestamp
        last_gripper_data = self.gripper.get_all_state()
        # last_gripper_data['gripper_position'] = last_gripper_data['gripper_position'] * 0.001   # 07/31

        # 100 Hz, tactile_receive_timestamp # 07/28
        last_tactile_data = self.tactile_listener.get_wrench()

        last_timestamp = self.last_camera_data[self.align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        # camera, robot, gripper data를 timestamp에 맞게 동기화 
        # align camera obs timestamps
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items(): # camera가 한대이므로 camera_idx = 0 
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                this_idxs.append(nn_idx)
            # remap key
            if camera_idx == 0 and self.mirror_crop:
                camera_obs['camera0_rgb'] = value['color'][...,:3][this_idxs]
                camera_obs['camera0_rgb_mirror_crop'] = value['color'][...,3:][this_idxs]
            else:
                camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        robot_pose_interpolator = PoseInterpolator(
            t=last_robot_data['robot_timestamp'], 
            x=last_robot_data['ActualTCPPose'])
        robot_pose = robot_pose_interpolator(robot_obs_timestamps)
        robot_obs = {
            'robot0_eef_pos': robot_pose[...,:3],
            'robot0_eef_rot_axis_angle': robot_pose[...,3:]
        }

        # align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        gripper_interpolator = get_interp1d(
            t=last_gripper_data['gripper_timestamp'],
            x=last_gripper_data['gripper_position'][...,None]
        )
        gripper_obs = {
            'robot0_gripper_width': gripper_interpolator(gripper_obs_timestamps)
        }

        # allign tactile obs
        # 07/28
        tactile_obs_horizon =2
        tactile_down_sample_steps = 1
        tactile_obs_timestamps = last_timestamp - (
            np.arange(tactile_obs_horizon)[::-1] * tactile_down_sample_steps * dt)
        # logger.debug(f"[DEBUG] last_tactile_data: {last_tactile_data}")
        # logger.debug(f"[DEBUG] type of each item: {[type(x) for x in last_tactile_data]}")
        # 디버깅 출력
        timestamps, taxels_array = last_tactile_data

        # for i, (timestamp, taxels) in enumerate(zip(timestamps, taxels_array)):
        #     logger.info(f"[DEBUG] item {i}: timestamp={timestamp}, last_timestamp={last_timestamp} taxels={taxels}, len: {len(taxels)}")

        tactile_interpolator = get_interp1d(
            t=timestamps,
            x=taxels_array
        )

        # 08/07 3dtacdex##################################
        force_3d = tactile_interpolator(tactile_obs_timestamps)
        force_3d = force_3d.reshape(-1, 16, 3)  # (N, 16, 3)
        normal_force_3d = (force_3d-self.mean) / self.std  # (N, 16, 3
        tactile_point = self._forward_kinematics(coords_type="full",coords_space="base")
        tactile_point = np.array(tactile_point)
        tactile_point_xyz = tactile_point[0,:,:3]
        T = normal_force_3d.shape[0]
        tactile_point_xyz_expanded = np.repeat(tactile_point_xyz[None, :, :], T, axis=0) 
        matched_tactile_xyzfxfyfz = np.concatenate([tactile_point_xyz_expanded, normal_force_3d],axis=2)
        data_batch, _, _, _ = self._data_to_gnn_batch(matched_tactile_xyzfxfyfz, edge_type='four+sensor')
        device = "cpu"
        data_batch = data_batch.to(device)
        latent_output = self.encoder(data_batch.x, data_batch.edge_index)
        latent_output = latent_output.view(2, -1)   # (batch_size, feature_dim)
        # logger.debug(f"[DEBUG] latent_output shape: {latent_output.shape}")
        tactile_obs = {
            'camera0_force_offset': latent_output.cpu().numpy() 
        }

        # tactile_obs = {
        #     'camera0_tactile_offset': tactile_interpolator(tactile_obs_timestamps)
        # }
        # logger.debug(f"[DEBUG] tactile_obs_timestamps: {tactile_obs_timestamps}")
        # logger.debug(f"[DEBUG] tactile_obs: {tactile_obs}")
        # accumulate obs
        # 각 timestamp에 따라 robot pose, joint, vel, gripper width를 robot_timestamp, gripper_timestamp에 따라 축적 
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                data={
                    'robot0_eef_pose': last_robot_data['ActualTCPPose'],
                    'robot0_joint_pos': last_robot_data['ActualQ'],
                    'robot0_joint_vel': last_robot_data['ActualQd'],
                },
                timestamps=last_robot_data['robot_timestamp']
            )
            self.obs_accumulator.put(
                data={
                    'robot0_gripper_width': last_gripper_data['gripper_position'][...,None]
                },
                timestamps=last_gripper_data['gripper_timestamp']
            )

        # return obs
        # camera_obs_timestamps를 기준으로 camera, robot, gripper 정보 통합하여 return
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data.update(tactile_obs)    # 07/28
        obs_data['timestamp'] = camera_obs_timestamps

        return obs_data
    
    def get_extend_obs(self) -> dict: # dictionary 형태로 return
       
        assert self.is_ready
        # self._init_tactile_encoder()

        # get data
        # 60 Hz, camera_calibrated_timestamp
        # 최근 k개의 데이터(frame, timestampe) 수집하여 last_camera_data로 저장
        # camera_obs_horizon = 2, camera_down_sample_steps =1, frequency =20 , math.ceil(올림) 
        k = math.ceil(
            self.camera_obs_horizon * self.camera_down_sample_steps \
            * (60 / self.frequency))
        self.last_camera_data = self.camera.get(
            k=k, 
            out=self.last_camera_data)

        # 125/500 hz, robot_receive_timestamp
        # robot = FrankaInterpolationController
        last_robot_data = self.robot.get_all_state()    # 최대 k개의 target_pose 
        # both have more than n_obs_steps data

        # 30 hz, gripper_receive_timestamp
        last_gripper_data = self.gripper.get_all_state()
        # last_gripper_data['gripper_position'] = last_gripper_data['gripper_position'] * 0.001   # 07/31

        # 100 Hz, tactile_receive_timestamp # 07/28
        last_tactile_data = self.tactile_listener.get_wrench()

        last_timestamp = self.last_camera_data[self.align_camera_idx]['timestamp'][-1]
        dt = 1 / self.frequency

        # camera, robot, gripper data를 timestamp에 맞게 동기화 
        # align camera obs timestamps
        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_obs_horizon)[::-1] * self.camera_down_sample_steps * dt)
        camera_obs = dict()
        for camera_idx, value in self.last_camera_data.items(): # camera가 한대이므로 camera_idx = 0 
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in camera_obs_timestamps:
                nn_idx = np.argmin(np.abs(this_timestamps - t))
                this_idxs.append(nn_idx)
            # remap key
            if camera_idx == 0 and self.mirror_crop:
                camera_obs['camera0_rgb'] = value['color'][...,:3][this_idxs]
                camera_obs['camera0_rgb_mirror_crop'] = value['color'][...,3:][this_idxs]
            else:
                camera_obs[f'camera{camera_idx}_rgb'] = value['color'][this_idxs]

        # align robot obs
        robot_obs_timestamps = last_timestamp - (
            np.arange(self.robot_obs_horizon)[::-1] * self.robot_down_sample_steps * dt)
        robot_pose_interpolator = PoseInterpolator(
            t=last_robot_data['robot_timestamp'], 
            x=last_robot_data['ActualTCPPose'])
        robot_pose = robot_pose_interpolator(robot_obs_timestamps)
        robot_obs = {
            'robot0_eef_pos': robot_pose[...,:3],
            'robot0_eef_rot_axis_angle': robot_pose[...,3:]
        }

        # align gripper obs
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_obs_horizon)[::-1] * self.gripper_down_sample_steps * dt)
        gripper_interpolator = get_interp1d(
            t=last_gripper_data['gripper_timestamp'],
            x=last_gripper_data['gripper_position'][...,None]
        )
        gripper_obs = {
            'robot0_gripper_width': gripper_interpolator(gripper_obs_timestamps)
        }

        # allign tactile obs
        # 07/28
        tactile_obs_horizon =2
        tactile_down_sample_steps = 1
        tactile_obs_timestamps = last_timestamp - (
            np.arange(tactile_obs_horizon)[::-1] * tactile_down_sample_steps * dt)
        # logger.debug(f"[DEBUG] last_tactile_data: {last_tactile_data}")
        # logger.debug(f"[DEBUG] type of each item: {[type(x) for x in last_tactile_data]}")
        # 디버깅 출력
        timestamps, taxels_array = last_tactile_data

        # for i, (timestamp, taxels) in enumerate(zip(timestamps, taxels_array)):
        #     logger.info(f"[DEBUG] item {i}: timestamp={timestamp}, last_timestamp={last_timestamp} taxels={taxels}, len: {len(taxels)}")

        tactile_interpolator = get_interp1d(
            t=timestamps,
            x=taxels_array
        )

        # 08/07 3dtacdex##################################
        force_3d = tactile_interpolator(tactile_obs_timestamps)
        force_3d = force_3d.reshape(-1, 16, 3)  # (N, 16, 3)\
        F = force_3d.sum(axis=1)  # sum over taxels

        tactile_obs = {
            'camera0_tactile_offset': F
        }
        # logger.debug(f"[DEBUG] tactile_obs_timestamps: {tactile_obs_timestamps}")
        # logger.debug(f"[DEBUG] tactile_obs: {tactile_obs}")
        # accumulate obs
        # 각 timestamp에 따라 robot pose, joint, vel, gripper width를 robot_timestamp, gripper_timestamp에 따라 축적 
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                data={
                    'robot0_eef_pose': last_robot_data['ActualTCPPose'],
                    'robot0_joint_pos': last_robot_data['ActualQ'],
                    'robot0_joint_vel': last_robot_data['ActualQd'],
                },
                timestamps=last_robot_data['robot_timestamp']
            )
            self.obs_accumulator.put(
                data={
                    'robot0_gripper_width': last_gripper_data['gripper_position'][...,None]
                },
                timestamps=last_gripper_data['gripper_timestamp']
            )

        # return obs
        # camera_obs_timestamps를 기준으로 camera, robot, gripper 정보 통합하여 return
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data.update(gripper_obs)
        obs_data.update(tactile_obs)    # 07/28
        obs_data['timestamp'] = camera_obs_timestamps

        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        # convert action to pose
        # receive_time을 기준으로, 그 이후의 action만 선택 
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        # compensate_latency가 true이면 robot, gripper action 보정 
        r_latency = self.robot_action_latency if compensate_latency else 0.0
        g_latency = self.gripper_action_latency if compensate_latency else 0.0

        # schedule waypoints
        for i in range(len(new_actions)):
            r_actions = new_actions[i,:6]   # 6D pose
            g_actions = new_actions[i,6:]   # 1D gripper_action 
            self.robot.schedule_waypoint(
                pose=r_actions,
                target_time=new_timestamps[i]-r_latency
            )
            self.gripper.schedule_waypoint(
                pos=g_actions,
                target_time=new_timestamps[i]-g_latency
            )

        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )


    def rdp_exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray,
            compensate_latency=False):

        # run_id = self._exec_run_id
        # self._exec_run_id += 1

        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        
        
        receive_time = time.time()
        margin = 0.02
        # logger.debug(f"Receive time: {receive_time}, Timestamps: {timestamps}")
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]

        # compensate_latency가 true이면 robot, gripper action 보정 
        # compensate_latency = 0.1
        self.robot_action_latency = 0.0 # 08/07
        self.gripper_action_latency = -1.5
        # logger.info(f"Compensate latency: {self.robot_action_latency} ")
        # logger.info(f"gripper latency: {self.gripper_action_latency}")
        r_latency = self.robot_action_latency if compensate_latency else 0.0
        # r_latency = 0.0
        g_latency = self.gripper_action_latency if compensate_latency else 0.0
        # r_latency = -0.05 # 08/07 special issue
        # r_latency = 2 * 0.1
        # g_latency = 2 * 0.1 
        r_ts = new_timestamps - r_latency 
        g_ts = new_timestamps - g_latency 
        # logger.debug(f"r_ts1: {r_ts}, g_ts: {g_ts}")
        # logger.info(f"keep time: {max(self._last_sched_wall_time_robot + margin, time.time() + margin)}")
        with self._wp_lock:
            group_id = self._wp_group_id
            self._wp_group_id += 1

            eps = 1e-3
            keep = r_ts > max(self._last_sched_wall_time_robot + eps, time.time() + margin)
            new_actions = new_actions[keep]
            # logger.debug(f"r_ts2: {r_ts}, g_ts: {g_ts}, keep: {keep}")
            r_ts = r_ts[keep]
            g_ts = g_ts[keep]
            # logger.debug(f"new_actions.shape: {new_actions.shape}, r_ts: {r_ts.shape}, g_ts: {g_ts.shape}")
            # logger.debug(f"last_sched_wall_time_robot: {self._last_sched_wall_time_robot} ")
            # logger.debug(f"r_ts: {r_ts}")

            for i in range(len(new_actions)):
                r_actions = new_actions[i,:6]   # 6D pose
                g_actions = new_actions[i,6:]   # 1D gripper_action 
                self.robot.schedule_waypoint(
                    pose=r_actions,
                    target_time=r_ts[i]
                )
                self.gripper.schedule_waypoint(    # 08/07 gripper action X for debug
                    pos=g_actions,
                    target_time=g_ts[i]
                )
                self.wp_logger.add(r_ts[i], r_actions, "robot",   group=group_id)
                self.wp_logger.add(g_ts[i], g_actions, "gripper", group=group_id)

            if r_ts.size > 0:
                self._last_sched_wall_time_robot = float(r_ts[-1])
            if g_ts.size > 0:
                self._last_sched_wall_time_grip = float(g_ts[-1])
        # if self.action_accumulator is not None and new_timestamps.size > 0:
        #     self.action_accumulator.put(new_actions, new_timestamps)
    
    def get_robot_state(self):
        return self.robot.get_state()
    
    def get_gripper_state(self):
        return self.gripper.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.camera.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on camera
        self.camera.restart_put(start_time=start_time)
        self.camera.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = ObsAccumulator()
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.camera.stop_recording()

        # TODO
        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            end_time = float('inf')
            for key, value in self.obs_accumulator.timestamps.items():
                end_time = min(end_time, value[-1])
            end_time = min(end_time, self.action_accumulator.timestamps[-1])

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            n_steps = 0
            if np.sum(self.action_accumulator.timestamps <= end_time) > 0:
                n_steps = np.nonzero(self.action_accumulator.timestamps <= end_time)[0][-1]+1

            if n_steps > 0:
                timestamps = action_timestamps[:n_steps]
                episode = {
                    'timestamp': timestamps,
                    'action': actions[:n_steps],
                }
                robot_pose_interpolator = PoseInterpolator(
                    t=np.array(self.obs_accumulator.timestamps['robot0_eef_pose']),
                    x=np.array(self.obs_accumulator.data['robot0_eef_pose'])
                )
                robot_pose = robot_pose_interpolator(timestamps)
                episode['robot0_eef_pos'] = robot_pose[:,:3]
                episode['robot0_eef_rot_axis_angle'] = robot_pose[:,3:]
                joint_pos_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_pos']),
                    np.array(self.obs_accumulator.data['robot0_joint_pos'])
                )
                joint_vel_interpolator = get_interp1d(
                    np.array(self.obs_accumulator.timestamps['robot0_joint_vel']),
                    np.array(self.obs_accumulator.data['robot0_joint_vel'])
                )
                episode['robot0_joint_pos'] = joint_pos_interpolator(timestamps)
                episode['robot0_joint_vel'] = joint_vel_interpolator(timestamps)

                gripper_interpolator = get_interp1d(
                    t=np.array(self.obs_accumulator.timestamps['robot0_gripper_width']),
                    x=np.array(self.obs_accumulator.data['robot0_gripper_width'])
                )
                episode['robot0_gripper_width'] = gripper_interpolator(timestamps)

                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')

    def _get_tactile_points(self,tactile_ori, tactile_points, link_pose, coords_type, coords_space):
        from scipy.spatial.transform import Rotation as R  # 안전: CUDA와 무관
        if coords_type == "full":
            local_points = tactile_ori + tactile_points # 각 taxel 상대 좌표 + 센서 원점
            rotation = link_pose[:3, :3]    
            translation = link_pose[:3, 3]
            real_points = (rotation @ local_points.T).T + translation   # 센서 기준 taxel 좌표를 로봇 base frame으로 변환 여기서는 link pose 고정이라 생각해 무시
            real_angle = R.from_matrix(rotation).as_euler("xyz").reshape(1, 3) / np.pi
            real_points = np.concatenate(
                [real_points, np.repeat(real_angle, self._point_per_sensor, axis=0)], axis=1
            )   # [16,6] (x,y,z,roll,pitch,yaw) -> 상대좌표를 사용하기 있으므로 taxel 위치 정보만 사용된다고 생각하면 될 듯 
        elif coords_type == "original": # used 
            local_points = np.array([tactile_ori])
            rotation = link_pose[:3, :3]
            translation = link_pose[:3, 3]
            real_points = (rotation @ local_points.T).T + translation
            real_angle = R.from_matrix(rotation).as_euler("xyz").reshape(1, 3) / np.pi

            if coords_space == "canonical": 
                min_point = np.min(tactile_points, axis=0)
                max_point = np.max(tactile_points, axis=0)
                diagonal_length = np.linalg.norm(max_point - min_point) # 가장 먼 taxel 간 거리
                center_point = (min_point + max_point) / 2
                tactile_points = 2 * (tactile_points - center_point) / diagonal_length  # taxel 좌표를 [-1,1] 범위로 정규화
                real_points = np.concatenate(
                    [
                        np.repeat(real_points, self._point_per_sensor, axis=0),
                        np.repeat(real_angle, self._point_per_sensor, axis=0),
                        tactile_points,
                    ],
                    axis=1,
                )
        return real_points
        
    def _forward_kinematics(self,coords_type="full",coords_space="base",):
        link_pose = np.eye(4)   # 센서가 고정된 pose로 가정
        coords_space = coords_space
        tactile_points = self._get_tactile_points(
            self._XELA_TACTILE_ORI_COORD,
            self._XELA_USPA44_COORD,
            link_pose,
            coords_type,
            coords_space,
        )
        return [tactile_points]

    def plot_trajectory(self, source='actual', smooth=False, dt=0.01, block=True):
        """
        source: 'actual' | 'waypoint'
        - 'actual': 로봇이 보고한 실제 EE 궤적 (get_all_state)
        - 'waypoint': 스케줄된 목표 포즈(로거에 기록된 pose)
        smooth: True면 인터폴레이터로 일정 간격 샘플링해서 매끈하게 그림
        dt: smooth 샘플 간격(초)
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # 1) 데이터 가져오기
        if source == 'actual':
            state = self.robot.get_all_state()
            ts = np.asarray(state['robot_timestamp'])
            poses = np.asarray(state['ActualTCPPose'])   # shape (N,6) [x,y,z,rx,ry,rz]
            label_prefix = 'Actual'
        elif source == 'waypoint':
            ts, poses = self.wp_logger.to_arrays(kind="robot")
            if ts.size == 0:
                print("[plot_trajectory] no waypoint yet."); return
            idx = np.argsort(ts); ts = ts[idx]; poses = poses[idx]
            label_prefix = 'Waypoint'
        else:
            raise ValueError("source must be 'actual' or 'waypoint'")

        # 2) 필요한 경우 매끈하게 보간
        if smooth and ts.size >= 2:
            # PoseInterpolator: (t, x) -> x(t)
            interp = PoseInterpolator(t=ts, x=poses)
            t0, t1 = float(ts[0]), float(ts[-1])
            ts_s = np.arange(t0, t1 + 1e-9, dt)
            poses_s = interp(ts_s)
            ts, poses = ts_s, poses_s

        # 3) 플롯 (XYZ / rotvec)
        x,y,z = poses[:,0], poses[:,1], poses[:,2]
        rx,ry,rz = poses[:,3], poses[:,4], poses[:,5]

        # time-series (pos)
        plt.figure()
        plt.plot(ts, x, label='x'); plt.plot(ts, y, label='y'); plt.plot(ts, z, label='z')
        plt.xlabel('time (s)'); plt.ylabel('pos (m)')
        plt.title(f'{label_prefix} EE Position'); plt.legend(); plt.tight_layout()

        # time-series (rotvec)
        plt.figure()
        plt.plot(ts, rx, label='rx'); plt.plot(ts, ry, label='ry'); plt.plot(ts, rz, label='rz')
        plt.xlabel('time (s)'); plt.ylabel('rotvec (rad)')
        plt.title(f'{label_prefix} EE Rotation (Rodrigues)'); plt.legend(); plt.tight_layout()

        # 3D 경로
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, marker='o')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title(f'{label_prefix} EE Path (XYZ)')
        plt.tight_layout(); plt.show(block=block)

    def plot_waypoint_smoothed(self, dt=0.005, block = True):
        # ts_wp, poses_wp = self.wp_logger.to_arrays(kind="robot")
        ts_wp, X_wp = self.wp_logger.to_arrays(kind="robot")
        ts_sc, X_sc = self.wp_logger.to_arrays(kind="robot_sched")
        # if ts_wp.size == 0:
        #     print("No waypoint."); return

        # idx = np.argsort(ts_wp)
        # t = ts_wp[idx].astype(float)
        # X = poses_wp[idx].astype(float)
        # keep = np.ones_like(t, dtype=bool)
        # keep[1:] = (np.diff(t) > 1e-6)
        # t = t[keep]; X = X[keep]

        # fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
        # if t.size >= 2:
        #     interp = PoseInterpolator(t=t, x=X)
        #     tq = np.arange(t[0], t[-1] + 1e-9, dt)
        #     Xq = interp(tq)
        #     ax.plot(Xq[:,0], Xq[:,1], Xq[:,2])   # 매끈한 라인

        # ax.scatter(X[:,0], X[:,1], X[:,2], s=8)  # 원래 waypoint 점
        # ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        # ax.set_title('Waypoint EE Path (interpolated line + points)')
        # self._set_axes_equal(ax)                 # ✅ self. 로 호출
        # plt.tight_layout(); plt.show(block=block)

        fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')

        # 선 먼저
        if X_sc.size > 0:
            ax.plot(X_sc[:,0], X_sc[:,1], X_sc[:,2], label='scheduled', linewidth=1)

        # 점 오버레이
        if X_wp.size > 0:
            ax.scatter(X_wp[:,0], X_wp[:,1], X_wp[:,2], s=10, label='waypoint')

        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title('Waypoint vs Scheduled Path')
        ax.legend()
        self._set_axes_equal(ax)
        plt.tight_layout(); plt.show(block=block)

    @staticmethod
    def _set_axes_equal(ax):
        xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
        xmid = np.mean(xlim); ymid = np.mean(ylim); zmid = np.mean(zlim)
        r = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) / 2
        ax.set_xlim3d([xmid-r, xmid+r])
        ax.set_ylim3d([ymid-r, ymid+r])
        ax.set_zlim3d([zmid-r, zmid+r])

    def _harvest_planned_samples(self):
        try:
            planned = self.robot.get_planned()
            if planned is None or len(planned) == 0:
                return
            ts = np.asarray(planned['planned_time'])
            X  = np.asarray(planned['planned_pose'])
            # 필요하면 정렬
            if ts.size > 0:
                idx = np.argsort(ts)
                ts, X = ts[idx], X[idx]
                for t, p in zip(ts, X):
                    self.wp_logger.add(t, p, "robot_sched")
        except Exception as e:
            logger.warning(f"harvest planned failed: {e}")


    def plot_waypoints_colored(self, kind="robot", smooth=True, dt=0.01, block=True,
                           max_legend_groups=8, legend_outside=True, use_colorbar=False,
                           figsize=(7,5)):
        """
        그룹(=exec 호출 묶음)별로 색을 다르게 표시.
        - use_colorbar=True: 범례를 없애고 컬러바로 그룹ID 표시(많은 그룹에 유리)
        - max_legend_groups: 범례에 최대 몇 개 그룹만 올릴지(나머지는 라벨 없음)
        - legend_outside=True: 범례를 축 밖(오른쪽)으로 빼서 플롯 영역 보존
        """
        import numpy as np
        import matplotlib.pyplot as plt

        ts, X, groups = self.wp_logger.to_arrays(kind=kind, return_groups=True)
        if ts.size == 0:
            print("[plot_waypoints_colored] no waypoint."); return

        # 시간순 정렬
        idx = np.argsort(ts)
        ts, X, groups = ts[idx], X[idx], groups[idx]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        uniq = np.unique(groups)

        if use_colorbar:
            # ✅ 컬러바 모드: 범례 없이도 많은 그룹을 깔끔하게 표현
            cmap = plt.get_cmap('turbo')   # 취향에 따라 'viridis', 'plasma' 등
            norm = plt.Normalize(vmin=uniq.min(), vmax=uniq.max())
            for g in uniq:
                m = (groups == g)
                tg, Xg = ts[m], X[m]
                color = cmap(norm(g))
                ax.scatter(Xg[:,0], Xg[:,1], Xg[:,2], s=12, color=color)
                if Xg.shape[0] >= 2:
                    # 타임스탬프 중복 제거(보간기 안전)
                    keep = np.ones_like(tg, dtype=bool)
                    keep[1:] = (np.diff(tg) > 1e-6)
                    tg = tg[keep]; Xg = Xg[keep]
                    if smooth and tg.size >= 2:
                        interp = PoseInterpolator(t=tg, x=Xg)
                        tq = np.arange(tg[0], tg[-1] + 1e-9, dt)
                        Xq = interp(tq)
                        ax.plot(Xq[:,0], Xq[:,1], Xq[:,2], color=color, linewidth=1.3)
                    elif Xg.shape[0] >= 2:
                        ax.plot(Xg[:,0], Xg[:,1], Xg[:,2], color=color, linewidth=1.0, alpha=0.8)

            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
            cbar.set_label('waypoint group id')

        else:
            # ✅ Top-K만 범례에 표시, 나머지는 라벨 없이 색만
            cmap = plt.get_cmap('tab20')   # 20색 반복
            # 크기(포인트 수) 기준으로 상위 그룹만 라벨
            sizes = [(g, np.sum(groups==g)) for g in uniq]
            sizes.sort(key=lambda x: x[1], reverse=True)
            legend_groups = set([g for g,_ in sizes[:max_legend_groups]])

            for k, g in enumerate(uniq):
                m = (groups == g)
                tg, Xg = ts[m], X[m]
                color = cmap(k % 20)
                label = f"group {g}" if g in legend_groups else None

                ax.scatter(Xg[:,0], Xg[:,1], Xg[:,2], s=12, color=color, label=label)
                if Xg.shape[0] >= 2:
                    keep = np.ones_like(tg, dtype=bool)
                    keep[1:] = (np.diff(tg) > 1e-6)
                    tg = tg[keep]; Xg = Xg[keep]
                    if smooth and tg.size >= 2:
                        interp = PoseInterpolator(t=tg, x=Xg)
                        tq = np.arange(tg[0], tg[-1] + 1e-9, dt)
                        Xq = interp(tq)
                        ax.plot(Xq[:,0], Xq[:,1], Xq[:,2], color=color, linewidth=1.3)
                    elif Xg.shape[0] >= 2:
                        ax.plot(Xg[:,0], Xg[:,1], Xg[:,2], color=color, linewidth=1.0, alpha=0.8)

            if max_legend_groups > 0:
                if legend_outside:
                    # 범례를 오른쪽 바깥으로
                    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                            borderaxespad=0., fontsize=8, frameon=False)
                    # 오른쪽 여백 확보
                    plt.tight_layout(rect=[0,0,0.80,1])
                else:
                    ax.legend(fontsize=8, frameon=False)

        # 축 설정
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title(f'Waypoints by group ({kind})')

        # 축 비율 1:1:1
        def _set_axes_equal(ax):
            xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
            xmid = np.mean(xlim); ymid = np.mean(ylim); zmid = np.mean(zlim)
            r = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) / 2
            ax.set_xlim3d([xmid-r, xmid+r])
            ax.set_ylim3d([ymid-r, ymid+r])
            ax.set_zlim3d([zmid-r, zmid+r])
        _set_axes_equal(ax)

        plt.tight_layout()
        plt.show(block=block)

    def plot_waypoints_colored_with_actual(self, kind="robot", smooth=True, dt=0.01, block=True,
                           max_legend_groups=8, legend_outside=True, use_colorbar=False,
                           figsize=(7,5)):
        """
        - WaypointLogger에 기록된 계획된 궤적(waypoints)을 그룹별로 색을 달리 표시
        - 동시에 실제 로봇이 보고한 ActualTCPPose도 함께 시각화
        """

        # ==== 1) Waypoints 불러오기 ====
        ts, X, groups = self.wp_logger.to_arrays(kind=kind, return_groups=True)
    
        has_wp = ts.size > 0
        if not has_wp:
            print("[plot_waypoints_colored_with_actual] no waypoint recorded.")

        # 시간순 정렬
        if has_wp:
            idx = np.argsort(ts)
            ts, X, groups = ts[idx], X[idx], groups[idx]

        # ==== 2) 실제 로봇 pos 불러오기 ====
        state = self.robot.get_all_state()
        ts_act = np.asarray(state['robot_timestamp'])
        X_act  = np.asarray(state['ActualTCPPose'])   # (N,6)
        has_actual = ts_act.size > 0

        # ==== 3) 플롯 ====
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # --- Waypoints (계획된 pose) ---
        if has_wp:
            uniq = np.unique(groups)
            if use_colorbar:
                cmap = plt.get_cmap('turbo')
                norm = plt.Normalize(vmin=uniq.min(), vmax=uniq.max())
                for g in uniq:
                    m = (groups == g)
                    tg, Xg = ts[m], X[m]
                    color = cmap(norm(g))
                    ax.scatter(Xg[:,0], Xg[:,1], Xg[:,2], s=12, color=color)
                    if Xg.shape[0] >= 2:
                        keep = np.ones_like(tg, dtype=bool)
                        keep[1:] = (np.diff(tg) > 1e-6)
                        tg, Xg = tg[keep], Xg[keep]
                        if smooth and tg.size >= 2:
                            interp = PoseInterpolator(t=tg, x=Xg)
                            tq = np.arange(tg[0], tg[-1] + 1e-9, dt)
                            Xq = interp(tq)
                            ax.plot(Xq[:,0], Xq[:,1], Xq[:,2], color=color, linewidth=1.3)
                        else:
                            ax.plot(Xg[:,0], Xg[:,1], Xg[:,2], color=color, linewidth=1.0, alpha=0.8)
                mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
                cbar = fig.colorbar(mappable, ax=ax, pad=0.02)
                cbar.set_label('waypoint group id')
            else:
                cmap = plt.get_cmap('tab20')
                sizes = [(g, np.sum(groups==g)) for g in uniq]
                sizes.sort(key=lambda x: x[1], reverse=True)
                legend_groups = set([g for g,_ in sizes[:max_legend_groups]])

                for k, g in enumerate(uniq):
                    m = (groups == g)
                    tg, Xg = ts[m], X[m]
                    color = cmap(k % 20)
                    label = f"group {g}" if g in legend_groups else None
                    ax.scatter(Xg[:,0], Xg[:,1], Xg[:,2], s=12, color=color, label=label)
                    if Xg.shape[0] >= 2:
                        keep = np.ones_like(tg, dtype=bool)
                        keep[1:] = (np.diff(tg) > 1e-6)
                        tg, Xg = tg[keep], Xg[keep]
                        if smooth and tg.size >= 2:
                            interp = PoseInterpolator(t=tg, x=Xg)
                            tq = np.arange(tg[0], tg[-1] + 1e-9, dt)
                            Xq = interp(tq)
                            ax.plot(Xq[:,0], Xq[:,1], Xq[:,2], color=color, linewidth=1.3)
                        else:
                            ax.plot(Xg[:,0], Xg[:,1], Xg[:,2], color=color, linewidth=1.0, alpha=0.8)

                if max_legend_groups > 0:
                    if legend_outside:
                        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                                borderaxespad=0., fontsize=8, frameon=False)
                        plt.tight_layout(rect=[0,0,0.80,1])
                    else:
                        ax.legend(fontsize=8, frameon=False)
            # Start : lime , End : red
            ax.scatter(X[0,0], X[0,1], X[0,2], color="green", s=50, marker="o", label="Waypoint Start")
            ax.scatter(X[-1,0], X[-1,1], X[-1,2], color="red", s=50, marker="X", label="Waypoint End")


        # --- Actual (실제 로봇 pos) ---
        if has_actual:
            ax.plot(X_act[:,0], X_act[:,1], X_act[:,2],
                    color='black', linewidth=2.0, alpha=0.6, label="Actual Trajectory")
            ax.scatter(X_act[:,0], X_act[:,1], X_act[:,2], color='black', s=8)

            # 실제 시작/끝점, Start : lime , End : red
            ax.scatter(X_act[0,0], X_act[0,1], X_act[0,2], color="lime", s=60, marker="^", label="Actual Start")
            ax.scatter(X_act[-1,0], X_act[-1,1], X_act[-1,2], color="red", s=60, marker="s", label="Actual End")
        # 축 설정
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title(f'Waypoints vs Actual(black) ({kind})')

        # 축 비율 1:1:1
        def _set_axes_equal(ax):
            xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
            xmid, ymid, zmid = np.mean(xlim), np.mean(ylim), np.mean(zlim)
            r = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) / 2
            ax.set_xlim3d([xmid-r, xmid+r])
            ax.set_ylim3d([ymid-r, ymid+r])
            ax.set_zlim3d([zmid-r, zmid+r])
        _set_axes_equal(ax)

        plt.tight_layout()
        plt.show(block=block)

    def plot_waypoints_vs_actual_aligned(self, kind="robot", dt=0.01, block=True, figsize=(7,5)):
        logger.info(f"[aligned] Plotting waypoints vs actual, kind={kind}, dt={dt}")
        ts_wp, X_wp, _ = self.wp_logger.to_arrays(kind=kind, return_groups=True)
        if ts_wp.size < 2:
            logger.info(f"[aligned] Not enough waypoint data."); return
        ts_act = np.asarray(self.robot.get_all_state()['robot_timestamp'])
        X_act  = np.asarray(self.robot.get_all_state()['ActualTCPPose'])
        if ts_act.size < 2:
            logger.info(f"[aligned] Not enough actual data."); return

        t_min = max(ts_wp.min(), ts_act.min())
        t_max = min(ts_wp.max(), ts_act.max())
        if t_min >= t_max:
            logger.info(f"[aligned] No overlap in time ranges: wp=({ts_wp.min():.2f},{ts_wp.max():.2f}), act=({ts_act.min():.2f},{ts_act.max():.2f})")
            return

        ts_common = np.arange(t_min, t_max, dt)
        logger.info(f"[aligned] using {len(ts_common)} common timestamps from {t_min:.2f} to {t_max:.2f}")

        interp_wp = PoseInterpolator(t=ts_wp, x=X_wp)
        interp_act = PoseInterpolator(t=ts_act, x=X_act)
        X_wp_s  = interp_wp(ts_common)
        X_act_s = interp_act(ts_common)

        # txt 저장
        # 절대 시간 대신 상대 시간으로 변환
        ts_rel = ts_common - ts_common[0]
        save_dir = "/home/embodied-ai/mcy"
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        from scipy.spatial.transform import Rotation as R  # 안전: CUDA와 무관
        quat_wp = R.from_rotvec(X_wp_s[:,3:]).as_quat()   # (N,4)
        quat_act = R.from_rotvec(X_act_s[:,3:]).as_quat() # (N,4)

        # wp_data = np.hstack([ts_common[:,None], X_wp_s[:,:3], quat_wp])  # (N, 8)
        act_data = np.hstack([ts_common[:,None], X_act_s[:,:3], quat_act])  # (N, 8)

        wp_data = np.hstack([ts_rel[:,None], X_wp_s[:,:3], quat_wp])
        act_data = np.hstack([ts_rel[:,None], X_act_s[:,:3], quat_act])

        # === txt 저장 ===
        np.savetxt(save_dir/"waypoints_aligned_quat.txt", wp_data,
                header="t x y z qx qy qz qw", fmt="%.6f")
        np.savetxt(save_dir/"actual_aligned_quat.txt", act_data,
                header="t x y z qx qy qz qw", fmt="%.6f")

        logger.info(f"[aligned] Saved txt files to {save_dir}")

        # error 분석
        err = X_wp_s[:,:6] - X_act_s[:,:6] # position error
        mean_err = np.mean(err, axis=0)
        rmse = np.sqrt(np.mean(np.square(err), axis=0))
        max_err = np.max(np.abs(err), axis=0)
        logger.info(f"[aligned] Position error (wp - act): mean={mean_err}, rmse={rmse}, max={max_err}")

        import matplotlib.pyplot as plt
        from matplotlib import cm
        cmap = cm.get_cmap("turbo")
        norm = plt.Normalize(vmin=ts_common.min(), vmax=ts_common.max())

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        for i, t in enumerate(ts_common):
            c = cmap(norm(t))
            ax.scatter(X_wp_s[i,0], X_wp_s[i,1], X_wp_s[i,2], color=c, marker="o", s=3)
            ax.scatter(X_act_s[i,0], X_act_s[i,1], X_act_s[i,2], color=c, marker="^", s=3)
            # logger.debug(f"[aligned] t={t:.3f}, wp=({X_wp_s[i,0]:.3f},{X_wp_s[i,1]:.3f},{X_wp_s[i,2]:.3f}), act=({X_act_s[i,0]:.3f},{X_act_s[i,1]:.3f},{X_act_s[i,2]:.3f})")
            # logger.debug(f"wp-act diff: ({X_wp_s[i,0]-X_act_s[i,0]:.3f},{X_wp_s[i,1]-X_act_s[i,1]:.3f},{X_wp_s[i,2]-X_act_s[i,2]:.3f})")
        logger.debug(f"mean wp-act diff: {np.mean(X_wp_s - X_act_s, axis=0)}")

        # 시작점 (공통 grid의 첫 번째 값)
        wp_start = X_wp_s[0]
        act_start = X_act_s[0]

        # 끝점 (공통 grid의 마지막 값)
        wp_end = X_wp_s[-1]
        act_end = X_act_s[-1]
        # 시작점 (waypoint =  green 동그라미, actual = 세모)
        ax.scatter(wp_start[0], wp_start[1], wp_start[2],
                color="green", s=60, marker="o", label="Waypoint Start")
        ax.scatter(act_start[0], act_start[1], act_start[2],
                color="green", s=60, marker="^", label="Actual Start")

        # 끝점 (waypoint = 빨간 동그라미, actual = 빨간 세모)
        ax.scatter(wp_end[0], wp_end[1], wp_end[2],
                color="red", s=60, marker="o", label="Waypoint End")
        ax.scatter(act_end[0], act_end[1], act_end[2],
                color="red", s=60, marker="^", label="Actual End")
        ax.set_title("Waypoint(circle) vs Actual(triangle) - green(start) red(end) (aligned)")

        # === 시간축 오차 플롯 ===
        fig2, ax2 = plt.subplots(figsize=(7,4))
        ax2.plot(ts_common, err[:,0], label="ex")
        ax2.plot(ts_common, err[:,1], label="ey")
        ax2.plot(ts_common, err[:,2], label="ez")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("pos error [m]")
        ax2.set_title(f"Position Error (Mean={mean_err}, RMSE={rmse}, Max={max_err})")
        ax2.legend()

        fig3, ax3 = plt.subplots(figsize=(7,4))
        ax3.plot(ts_common, err[:,3], label="erx")
        ax3.plot(ts_common, err[:,4], label="ery")
        ax3.plot(ts_common, err[:,5], label="erz")
        ax3.set_xlabel("time [s]")
        ax3.set_ylabel("pos error [m]")
        ax3.set_title(f"Position Error (Mean={mean_err}, RMSE={rmse}, Max={max_err})")
        ax3.legend()


        plt.show(block=block)

    def next_robot_time(self, margin: float = 0.02) -> float:
        with self._wp_lock:
            dt = 1.0 / self.frequency
            logger.info(f"next_robot_time: {self._last_sched_wall_time_robot + dt}, current time: {time.time() + margin}")
            return max(self._last_sched_wall_time_robot + dt, time.time() + margin)

    def next_gripper_time(self, margin: float = 0.02) -> float:
        with self._wp_lock:
            dt = 1.0 / self.frequency
            return max(self._last_sched_wall_time_grip + dt, time.time() + margin)