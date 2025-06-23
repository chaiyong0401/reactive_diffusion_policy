import threading
import time
import os.path as osp
import numpy as np
import torch
import tqdm
from loguru import logger
from typing import Dict, Tuple, Union, Optional
import rclpy
import transforms3d as t3d
import py_cli_interaction
from rclpy.executors import MultiThreadedExecutor
from omegaconf import DictConfig, ListConfig
from reactive_diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.common.precise_sleep import precise_sleep
from reactive_diffusion_policy.env.real_bimanual.real_env import RealRobotEnvironment
from reactive_diffusion_policy.real_world.real_inference_util import (
    get_real_obs_dict)
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
from reactive_diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix, pose10d_to_mat, mat_to_pose
from reactive_diffusion_policy.common.ensemble import EnsembleBuffer
from reactive_diffusion_policy.common.action_utils import (
    interpolate_actions_with_ratio,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions
)
import requests


import os
import psutil
from copy import deepcopy

# add this to prevent assigning too may threads when using numpy
# 각 library 별ㄹ 최대 12개 제한 
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
# add this to prevent assigning too may threads when using open-cv
cv2.setNumThreads(12)

# Get the total number of CPU cores
total_cores = psutil.cpu_count()
# Define the number of cores you want to bind to
num_cores_to_bind = 10
# Calculate the indices of the first ten cores
# Ensure the number of cores to bind does not exceed the total number of cores
cores_to_bind = set(range(min(num_cores_to_bind, total_cores)))
# Set CPU affinity for the current process to the first ten cores
os.sched_setaffinity(0, cores_to_bind)

class RealRunner:
    def __init__(self,
                 output_dir: str,
                 transform_params: DictConfig,
                 env_params: DictConfig,
                 shape_meta: DictConfig,
                 tcp_ensemble_buffer_params: DictConfig,
                 gripper_ensemble_buffer_params: DictConfig,
                 latent_tcp_ensemble_buffer_params: DictConfig = None,
                 latent_gripper_ensemble_buffer_params: DictConfig = None,
                 use_latent_action_with_rnn_decoder: bool = False,
                 use_relative_action: bool = False,
                 use_relative_tcp_obs_for_relative_action: bool = True,
                 action_interpolation_ratio: int = 1,
                 eval_episodes=10,
                 max_duration_time: float = 30,
                 tcp_action_update_interval: int = 6,
                 gripper_action_update_interval: int = 10,
                 tcp_pos_clip_range: ListConfig = ListConfig([[0.6, -0.4, 0.03], [1.0, 0.45, 0.4]]),
                #  tcp_rot_clip_range: ListConfig = ListConfig([[-np.pi, 0., np.pi], [-np.pi, 0., np.pi]]), # original 상수로 값 고정
                tcp_rot_clip_range: ListConfig = ListConfig([[-np.pi, -np.pi, -np.pi], [np.pi, np.pi, np.pi]]), # no clip
                 tqdm_interval_sec = 5.0,
                 control_fps: float = 12,
                 inference_fps: float = 6,
                 latency_step: int = 0,
                 gripper_latency_step: Optional[int] = None,
                 n_obs_steps: int = 2,  # 몇 개의 시점을 stack할지
                 obs_temporal_downsample_ratio: int = 2,    # obs 샘플간 간격 [02, 04,..]
                 dataset_obs_temporal_downsample_ratio: int = 1,
                 downsample_extended_obs: bool = True,
                 enable_video_recording: bool = False,
                 vcamera_server_ip: Optional[Union[str, ListConfig]] = None,
                 vcamera_server_port: Optional[Union[int, ListConfig]] = None,
                 task_name=None,
                 ):
        self.task_name = task_name
        self.transforms = RealWorldTransforms(option=transform_params)
        self.shape_meta = dict(shape_meta)
        self.eval_episodes = eval_episodes

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        logger.info(f"RGB_keys: {rgb_keys}")
        logger.info(f"lowdim_keys: {lowdim_keys}")

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                extended_rgb_keys.append(key)
            elif type == 'low_dim':
                extended_lowdim_keys.append(key)
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys
        logger.info(f"extended_RGB_keys: {extended_rgb_keys}")
        logger.info(f"extended_lowdim_keys:{extended_lowdim_keys}")

        rclpy.init(args=None)
        self.env = RealRobotEnvironment(transforms=self.transforms, **env_params)
        # set gripper to max width
        self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width)
        time.sleep(2)

        self.max_duration_time = max_duration_time
        self.tcp_action_update_interval = tcp_action_update_interval
        self.gripper_action_update_interval = gripper_action_update_interval
        self.tcp_pos_clip_range = tcp_pos_clip_range
        self.tcp_rot_clip_range = tcp_rot_clip_range
        self.tqdm_interval_sec = tqdm_interval_sec
        self.control_fps = control_fps
        self.control_interval_time = 1.0 / control_fps
        self.inference_fps = inference_fps
        self.inference_interval_time = 1.0 / inference_fps
        assert self.control_fps % self.inference_fps == 0
        self.latency_step = latency_step
        self.gripper_latency_step = gripper_latency_step if gripper_latency_step is not None else latency_step
        self.n_obs_steps = n_obs_steps
        self.obs_temporal_downsample_ratio = obs_temporal_downsample_ratio
        self.dataset_obs_temporal_downsample_ratio = dataset_obs_temporal_downsample_ratio
        self.downsample_extended_obs = downsample_extended_obs
        self.use_latent_action_with_rnn_decoder = use_latent_action_with_rnn_decoder
        if self.use_latent_action_with_rnn_decoder:
            assert latent_tcp_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action."
            assert latent_gripper_ensemble_buffer_params.ensemble_mode == 'new', "Only support new ensemble mode for latent action."
            self.tcp_ensemble_buffer = EnsembleBuffer(**latent_tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**latent_gripper_ensemble_buffer_params)
        else:
            self.tcp_ensemble_buffer = EnsembleBuffer(**tcp_ensemble_buffer_params)
            self.gripper_ensemble_buffer = EnsembleBuffer(**gripper_ensemble_buffer_params)
        self.use_relative_action = use_relative_action
        self.use_relative_tcp_obs_for_relative_action = use_relative_tcp_obs_for_relative_action
        self.action_interpolation_ratio = action_interpolation_ratio

        self.enable_video_recording = enable_video_recording
        if enable_video_recording:
            assert isinstance(vcamera_server_ip, str) and isinstance(vcamera_server_port, int) or \
                     isinstance(vcamera_server_ip, ListConfig) and isinstance(vcamera_server_port, ListConfig), \
                "vcamera_server_ip and vcamera_server_port should be a string or ListConfig."
        if isinstance(vcamera_server_ip, str):
            vcamera_server_ip_list = [vcamera_server_ip]
            vcamera_server_port_list = [vcamera_server_port]
        elif isinstance(vcamera_server_ip, ListConfig):
            vcamera_server_ip_list = list(vcamera_server_ip)
            vcamera_server_port_list = list(vcamera_server_port)
        self.vcamera_server_ip_list = vcamera_server_ip_list
        self.vcamera_server_port_list = vcamera_server_port_list
        self.video_dir = osp.join(output_dir, 'videos')

        self.stop_event = threading.Event()
        self.session = requests.Session()

    @staticmethod
    def spin_executor(executor):
        executor.spin()

    def pre_process_obs(self, obs_dict: Dict) -> Tuple[Dict, Dict]:
        obs_dict = deepcopy(obs_dict)

        for key in self.lowdim_keys:
            if "wrt" not in key:
                obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        # inter-gripper relative action
        obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        for key in self.lowdim_keys:
            obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]]

        absolute_obs_dict = dict()
        for key in self.lowdim_keys:
            absolute_obs_dict[key] = obs_dict[key].copy()

        # convert absolute action to relative action 
        # use mcy
        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            for key in self.lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = obs_dict[key][-1].copy() # 1 item (9d)
                    obs_dict[key] = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=base_absolute_action) # 2 items 9d
                    logger.debug(f"base absolute action in pre process obs: {base_absolute_action}")
                    logger.debug(f"relative action in pre process obs: {obs_dict[key]}")
        return obs_dict, absolute_obs_dict

    def pre_process_extended_obs(self, extended_obs_dict: Dict) -> Tuple[Dict, Dict]:
        extended_obs_dict = deepcopy(extended_obs_dict)

        absolute_extended_obs_dict = dict()
        for key in self.extended_lowdim_keys:
            extended_obs_dict[key] = extended_obs_dict[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]]
            absolute_extended_obs_dict[key] = extended_obs_dict[key].copy()

        logger.info(f"In pre_process_extended_obs")
        # convert absolute action to relative action -> not use in mcy
        # now in extended_lowdim_keys = left_gripper1_marker_offset_emb
        if self.use_relative_action and self.use_relative_tcp_obs_for_relative_action:
            logger.info(f"use self.use relative action")
            for key in self.extended_lowdim_keys:
                if 'robot_tcp_pose' in key and 'wrt' not in key:
                    base_absolute_action = extended_obs_dict[key][-1].copy()
                    extended_obs_dict[key] = absolute_actions_to_relative_actions(extended_obs_dict[key], base_absolute_action=base_absolute_action)
                    # logger.debug(f"base_absolute_action in preprocess_extended_obs: {base_absolute_action}")
                    # logger.debug(f"extended_obs_dict in preprocess_extended_obs: {extended_obs_dict[key]}")

        return extended_obs_dict, absolute_extended_obs_dict

    def post_process_action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Post-process the action before sending to the robot
        """
        assert len(action.shape) == 2  # (action_steps, d_a) 
        if self.env.data_processing_manager.use_6d_rotation:  # mcy use 6d rotation 
            # logger.info("we use_6d_rotation")
            if action.shape[-1] == 4 or action.shape[-1] == 8: # (x,y,z,grip) -> left_action_6d = (x,y,z,0,0,0)
                # convert to 6D pose
                left_trans_batch = action[:, :3]  # (action_steps, 3)
                # we use default euler angles as 0
                left_euler_batch = np.zeros_like(left_trans_batch)
                left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                if action.shape[-1] == 8:
                    right_trans_batch = action[:, 3:6]  # (action_steps, 3)
                    right_euler_batch = np.zeros_like(right_trans_batch)
                    right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                else:
                    right_action_6d = None
            # rot 6d -> rotation matrix -> Euler(3 angles)
            elif action.shape[-1] == 10 or action.shape[-1] == 20: # (x,y,z,rx1,rx2,rx3,ry1,ry2,ry3,gripper) -> (x,y,z,roll,pitch,yaw)
                # convert to 6D pose
                # left_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 3:9])  # (6d rotation을 matrix 형태로 변환 (umi와 유사)
                # left_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in left_rot_mat_batch])  # (action_steps, 3)
                # left_trans_batch = action[:, :3]  # (action_steps, 3) (x,y,z)
                # left_action_6d = np.concatenate([left_trans_batch, left_euler_batch], axis=1)  # (action_steps, 6)
                left_action_batch = pose10d_to_mat(action[:,:9])
                left_action_6d = mat_to_pose(left_action_batch)
                # left_action_6d[:,3] = left_action_6d[:,3]- np.pi/2
                if action.shape[-1] == 20:
                    right_rot_mat_batch = ortho6d_to_rotation_matrix(action[:, 12:18])
                    right_euler_batch = np.array([t3d.euler.mat2euler(rot_mat) for rot_mat in right_rot_mat_batch])
                    right_trans_batch = action[:, 9:12]
                    right_action_6d = np.concatenate([right_trans_batch, right_euler_batch], axis=1)
                else:
                    right_action_6d = None
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # clip action (x, y, z)
        logger.debug(f"action_rotation before clip: {left_action_6d[:,3:]}")
        left_action_6d[:, :3] = np.clip(left_action_6d[:, :3], np.array(self.tcp_pos_clip_range[0]), np.array(self.tcp_pos_clip_range[1]))
        if right_action_6d is not None:
            right_action_6d[:, :3] = np.clip(right_action_6d[:, :3], np.array(self.tcp_pos_clip_range[2]), np.array(self.tcp_pos_clip_range[3]))
        # clip action (r, p, y)
        ######## no clip left_action_6d
        # left_action_6d[:, 3:] = np.clip(left_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[0]), np.array(self.tcp_rot_clip_range[1]))
        if right_action_6d is not None:
            right_action_6d[:, 3:] = np.clip(right_action_6d[:, 3:], np.array(self.tcp_rot_clip_range[2]), np.array(self.tcp_rot_clip_range[3]))
        # add gripper action
        if action.shape[-1] == 4:
            left_action = np.concatenate([left_action_6d, action[:, 3][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif action.shape[-1] == 8:
            left_action = np.concatenate([left_action_6d, action[:, 6][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = np.concatenate([right_action_6d, action[:, 7][:, np.newaxis],
                                           np.zeros((action.shape[0], 1))], axis=1)
        elif action.shape[-1] == 10:
            left_action = np.concatenate([left_action_6d, action[:, 9][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = None
        elif action.shape[-1] == 20:
            left_action = np.concatenate([left_action_6d, action[:, 18][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)
            right_action = np.concatenate([right_action_6d, action[:, 19][:, np.newaxis],
                                          np.zeros((action.shape[0], 1))], axis=1)

        else:
            raise NotImplementedError

        if right_action is None:
            right_action = left_action.copy()
            is_bimanual = False
        else:
            is_bimanual = True
        logger.debug(f"left_action: {left_action}")
        action_all = np.concatenate([left_action, right_action], axis=-1)
        return (action_all, is_bimanual)

    # use_mcy 
    def action_command_thread(self, policy: Union[DiffusionUnetImagePolicy], stop_event):
        while not stop_event.is_set():  # stop event 발생하기 전까지 loop 진행 
            start_time = time.time()
            logger.info("action_command_thread")
            # get step action from ensemble buffer
            # tcp_ensemble_buffer, gripper_ensemble_buffer에서 각각 가장 최근(if ensemble mode = "new") 값 가져옴
            tcp_step_action = self.tcp_ensemble_buffer.get_action() 
            gripper_step_action = self.gripper_ensemble_buffer.get_action()
            if tcp_step_action is None or gripper_step_action is None:  # no action in the buffer => no movement.
                cur_time = time.time()
                precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
                logger.debug(f"Step: {self.action_step_count}, control_interval_time: {self.control_interval_time}, "
                             f"cur_time-start_time: {cur_time - start_time}")
                self.action_step_count += 1
                continue
            
            if self.use_latent_action_with_rnn_decoder: # mcy use 
                tcp_extended_obs_step = int(tcp_step_action[-1])    # tcp, gripper의 마지막 요소(step) 추출 
                gripper_extended_obs_step = int(gripper_step_action[-1])
                tcp_step_action = tcp_step_action[:-1]  # step 제외한 나머지 -> latent 
                gripper_step_action = gripper_step_action[:-1]

                longer_extended_obs_step = max(tcp_extended_obs_step, gripper_extended_obs_step) # 둘 중 더 긴 time step 사용 
                obs_temporal_downsample_ratio = self.obs_temporal_downsample_ratio if self.downsample_extended_obs else 1
                extended_obs = self.env.get_obs(longer_extended_obs_step,
                                                    temporal_downsample_ratio= obs_temporal_downsample_ratio)

                if self.use_relative_action: # mcy use 
                    action_dim = self.shape_meta['obs']['left_robot_tcp_pose']['shape'][0]
                    if 'right_robot_tcp_pose' in self.shape_meta['obs']:
                        action_dim += self.shape_meta['obs']['right_robot_tcp_pose']['shape'][0]
                    tcp_base_absolute_action = tcp_step_action[-action_dim:]
                    gripper_base_absolute_action = gripper_step_action[-action_dim:]
                    tcp_step_action = tcp_step_action[:-action_dim]
                    gripper_step_action = gripper_step_action[:-action_dim]

                np_extended_obs_dict = dict(extended_obs)
                np_extended_obs_dict = get_real_obs_dict(
                    env_obs=np_extended_obs_dict, shape_meta=self.shape_meta, is_extended_obs=True)
                np_extended_obs_dict, _ = self.pre_process_extended_obs(np_extended_obs_dict)
                extended_obs_dict = dict_apply(np_extended_obs_dict, lambda x: torch.from_numpy(x).unsqueeze(0))

                tcp_step_latent_action = torch.from_numpy(tcp_step_action.astype(np.float32)).unsqueeze(0)
                gripper_step_latent_action = torch.from_numpy(gripper_step_action.astype(np.float32)).unsqueeze(0)

                dataset_obs_temporal_downsample_ratio = self.dataset_obs_temporal_downsample_ratio
                tcp_step_action = policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
                gripper_step_action = policy.predict_from_latent_action(gripper_step_latent_action, extended_obs_dict, gripper_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
                if self.use_relative_action:
                    # logger.debug(f"tcp_step_action: {tcp_step_action}")
                    # logger.debug(f"tcp_base_absolute_action: {tcp_base_absolute_action}")
                    tcp_step_action = relative_actions_to_absolute_actions(tcp_step_action, tcp_base_absolute_action)
                    gripper_step_action = relative_actions_to_absolute_actions(gripper_step_action, gripper_base_absolute_action)
                    # logger.debug(f"tcp_relative_to_absolute_step_action: {tcp_step_action}")

                if tcp_step_action.shape[-1] == 4: # (x, y, z, gripper_width)
                    tcp_len = 3
                elif tcp_step_action.shape[-1] == 8: # (x_l, y_l, z_l, x_r, y_r, z_r, gripper_width_l, gripper_width_r)
                    tcp_len = 6
                elif tcp_step_action.shape[-1] == 10: # (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
                    tcp_len = 9
                elif tcp_step_action.shape[-1] == 20: # (x_l, y_l, z_l, rotation_l, x_r, y_r, z_r, rotation_r, gripper_width_l, gripper_width_r)
                    tcp_len = 18
                else:
                    raise NotImplementedError
                logger.debug(f"tcp_len = {tcp_len}")
                logger.debug(f"tcp_step_action_shape: {tcp_step_action.shape[-1]}")
                if self.env.enable_exp_recording:
                    self.env.get_predicted_action(tcp_step_action[:, :tcp_len], type='partial_tcp')
                    self.env.get_predicted_action(gripper_step_action[:, tcp_len:], type='partial_gripper')

                    full_tcp_step_action = policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, dataset_obs_temporal_downsample_ratio, extend_obs_pad_after=True)['action'][0].detach().cpu().numpy()
                    full_gripper_step_action = policy.predict_from_latent_action(gripper_step_latent_action, extended_obs_dict, gripper_extended_obs_step, dataset_obs_temporal_downsample_ratio, extend_obs_pad_after=True)['action'][0].detach().cpu().numpy()
                    if self.use_relative_action:
                        full_tcp_step_action = relative_actions_to_absolute_actions(full_tcp_step_action, tcp_base_absolute_action)
                        full_gripper_step_action = relative_actions_to_absolute_actions(full_gripper_step_action, gripper_base_absolute_action)
                    self.env.get_predicted_action(full_tcp_step_action[:, :tcp_len], type='full_tcp')
                    self.env.get_predicted_action(full_gripper_step_action[:, tcp_len:], type='full_gripper')

                tcp_step_action = tcp_step_action[-1]
                gripper_step_action = gripper_step_action[-1]
                logger.info(f"tcp_step_action: {tcp_step_action}")

                tcp_step_action = tcp_step_action[:tcp_len]
                gripper_step_action = gripper_step_action[tcp_len:]

            combined_action = np.concatenate([tcp_step_action, gripper_step_action], axis=-1)
            # convert to 16-D robot action (TCP(left,right 6D) + gripper of both arms(left, right gripper width, force))
            # TODO: handle rotation in temporal ensemble buffer!
            # is_bimanual : use one hand or bi-man
            # current gripper force is 0 (dummy) actually [TCP pose + gripper width] used 
            step_action, is_bimanual = self.post_process_action(combined_action[np.newaxis, :])
            step_action = step_action.squeeze(0)

            # send action to the robot
            print("excute_action")
            logger.info(f"step action = {step_action}")
            self.env.execute_action(step_action, use_relative_action=False, is_bimanual=is_bimanual) # env.excute_action에 use_relative_action 정의 안됨
            # with open(tcp_pose_log_path, 'a') as f:
            #                         f.write(f"{step_count},{np_absolute_obs_dict['left_robot_tcp_pose'][-1]}\n")
            cur_time = time.time()
            precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))
            self.action_step_count += 1

    def start_record_video(self, video_path):
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/start_recording/{video_path}')
            if response.status_code == 200:
                logger.info(f"Start recording video to {video_path}")
            else:
                logger.error(f"Failed to start recording video to {video_path}")

    def stop_record_video(self):
        for vcamera_server_ip, vcamera_server_port in zip(self.vcamera_server_ip_list, self.vcamera_server_port_list):
            response = self.session.post(f'http://{vcamera_server_ip}:{vcamera_server_port}/stop_recording')
            if response.status_code == 200:
                logger.info(f"Stop recording video")
            else:
                logger.error(f"Failed to stop recording video")

    # Main Thread (#1) #############################################################################
    def run(self, policy: Union[DiffusionUnetImagePolicy]):
        if self.use_latent_action_with_rnn_decoder: # Latent diffusion policy
            assert policy.at.use_rnn_decoder, "Policy should use rnn decoder for latent action."
        else:   
            assert not hasattr(policy, 'at') or not policy.at.use_rnn_decoder, "Policy should not use rnn decoder for action."

        device = policy.device

        executor = MultiThreadedExecutor()
        executor.add_node(self.env)

        try:
            # RealRobotEnvironment node를 비동기적으로 실행하는 thread spin (#2) #########################
            spin_thread = threading.Thread(target=self.spin_executor, args=(executor,), daemon=True)
            spin_thread.start()

            time.sleep(2)
            # progress bar를 보여주는 for loop 
            for episode_idx in tqdm.tqdm(range(0, self.eval_episodes),
                                         desc=f"Eval for {self.task_name}",
                                         leave=False, mininterval=self.tqdm_interval_sec):
                logger.info(f"Start evaluation episode {episode_idx}")
                # ask user whether the environment resetting is done
                tcp_pose_log_path = osp.join(self.video_dir, f"episode_{episode_idx}_tcp_pose.txt")
                action_log_path = osp.join(self.video_dir, f"episode_{episode_idx}_executed_actions.txt")
                reset_flag = py_cli_interaction.parse_cli_bool('Has the environment reset finished?', default_value=True)
                if not reset_flag:
                    logger.warning("Skip this episode.")
                    continue

                logger.info("Start episode rollout.")
                # start rollout
                self.env.reset()
                # set gripper to max width
                self.env.send_gripper_command_direct(self.env.max_gripper_width, self.env.max_gripper_width)
                time.sleep(1)

                policy.reset()
                self.tcp_ensemble_buffer.clear()
                self.gripper_ensemble_buffer.clear()
                logger.debug("Reset environment and policy.")

                if self.enable_video_recording:
                    video_path = os.path.join(self.video_dir, f'episode_{episode_idx}.mp4')
                    # self.start_record_video(video_path)
                    logger.info(f"Start recording video to {video_path}")

                self.stop_event.clear()
                time.sleep(0.5)
                # start a new thread for action command (#3) ###############################################
                action_thread = threading.Thread(target=self.action_command_thread, args=(policy, self.stop_event,),
                                                 daemon=True)
                action_thread.start()

                self.action_step_count = 0
                step_count = 0
                steps_per_inference = int(self.control_fps / self.inference_fps)    # 12 / 6 -> 2 
                start_timestamp = time.time()
                last_timestamp = start_timestamp
                try:
                    while True:
                        # profiler = Profiler()
                        # profiler.start()
                        start_time = time.time()
                        # get obs, 최근 n_obs_steps(2) 만큼의 obs를 temporal_downsample_ratio(1) 간격으로 획득 
                        obs = self.env.get_obs(
                            obs_steps=self.n_obs_steps,
                            temporal_downsample_ratio=self.obs_temporal_downsample_ratio)
                        # obs = dict()
                        # logger.debug(f"obs_key:{obs.keys()}")

                        if len(obs) == 0:
                            logger.warning("No observation received! Skip this step.")
                            cur_time = time.time()
                            precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time)))
                            step_count += steps_per_inference
                            continue

                        # create obs dict
                        np_obs_dict = dict(obs)
                        # get transformed real obs dict
                        # np_obs_dict_low_dim = 2*[tcp pose(9d),gripper_width(1d),marker_embedding(63)]
                        np_obs_dict = get_real_obs_dict(
                            env_obs=np_obs_dict, shape_meta=self.shape_meta)
                        np_obs_dict, np_absolute_obs_dict = self.pre_process_obs(np_obs_dict)

                        # device transfer
                        obs_dict = dict_apply(np_obs_dict,
                                              lambda x: torch.from_numpy(x).unsqueeze(0).to(
                                                  device=device))

                        policy_time = time.time()
                        # run policy
                        # observation에 들어있는 tcp_pose는 get_ee_pose로 얻은 6d data를 9d로 변환한 것이기 때문애 성대 위치가 아니다. 
                        with torch.no_grad():
                            if self.use_latent_action_with_rnn_decoder: # use_mcy
                                action_dict = policy.predict_action(obs_dict,
                                                                    dataset_obs_temporal_downsample_ratio=self.dataset_obs_temporal_downsample_ratio,
                                                                    return_latent_action=True)
                            else:
                                action_dict = policy.predict_action(obs_dict)
                        logger.debug(f"Policy inference time: {time.time() - policy_time:.3f}s")

                        # device_transfer
                        np_action_dict = dict_apply(action_dict,
                                                    lambda x: x.detach().to('cpu').numpy())

                        action_all = np_action_dict['action'].squeeze(0)
                        if self.use_latent_action_with_rnn_decoder: ## we use
                            # add first absolute action to get absolute action
                            if self.use_relative_action:
                                base_absolute_action = np.concatenate([ # 현재 시점의 absolute pose obs값, [-1]: 가장 최근 값 
                                    np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                                    np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                                ], axis=-1)
                                # with open(tcp_pose_log_path, 'a') as f:
                                #     f.write(f"{step_count},{np_absolute_obs_dict['left_robot_tcp_pose'][-1]}\n")

                                action_all = np.concatenate([   # action all = (action_all의 각 timesteps base_absolute_action) -> shape(num_timesteps, latent_dim + left_tcp_pose_dim + right_tcp_pose_dim)
                                    action_all, # shape(num_timesteps, latent_dim)
                                    base_absolute_action[np.newaxis, :].repeat(action_all.shape[0], axis=0) # base_absolute_action을 action_all.shape[0](num_timesteps) 만큼 복제 -> shape(num_timesteps, left_tcp_pose_dim + right_tcp_pose_dim)
                                ], axis=-1)
                            # add action step to get corresponding observation
                            # 각 latent action timestep이 이후에 들어올 어느 시점에 대응되는지 알려주는 timestamp ###########################################################
                            action_all = np.concatenate([  # action이 n_obs_steps 이후 timestep에 대응되므로, 이에 맞는 time index 부여 -> action all shape : [num_timestpes, latent_dim + left_tcp_pose_dim + right_tcp_pose_dim + num_timestep(1)]
                                action_all,
                                np.arange(self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio, action_all.shape[0] + self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio)[:, np.newaxis]    #(2, num_timesteps + 2)[:,np.newaxis] -> shape (num_timesteps,1)
                            ], axis=-1)
                        else: ## not use 
                            if self.use_relative_action:
                                base_absolute_action = np.concatenate([
                                    np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                                    np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                                ], axis=-1)
                                action_all = relative_actions_to_absolute_actions(action_all, base_absolute_action)

                        if self.action_interpolation_ratio > 1: # not use 
                            if self.use_latent_action_with_rnn_decoder:
                                action_all = action_all.repeat(self.action_interpolation_ratio, axis=0)
                            else:
                                action_all = interpolate_actions_with_ratio(action_all, self.action_interpolation_ratio)    

                        # TODO: only takes the first n_action_steps and add to the ensemble buffer
                        # action_all 중 일부를 TCP(latent)와 gripper에 나눠서 ensemble buffer에 넣기
                        if step_count % self.tcp_action_update_interval == 0:   # 현재 step이 TCP update 주기(6)에 도달했을 때 TCP action update
                            if self.use_latent_action_with_rnn_decoder:
                                tcp_action = action_all[self.latency_step:, ...]    # latency_step = 0 -> shape : (num_timesteps, D)
                                logger.debug(f"self_latency_step: {self.latency_step}")
                            else:   # not use 
                                if action_all.shape[-1] == 4:
                                    tcp_action = action_all[self.latency_step:, :3]
                                elif action_all.shape[-1] == 8:
                                    tcp_action = action_all[self.latency_step:, :6]
                                elif action_all.shape[-1] == 10:
                                    tcp_action = action_all[self.latency_step:, :9]
                                elif action_all.shape[-1] == 20:
                                    tcp_action = action_all[self.latency_step:, :18]
                                else:
                                    raise NotImplementedError
                            # add to ensemble buffer
                            logger.debug(f"Step: {step_count}, Add TCP action to ensemble buffer: {tcp_action}")
                            with open(action_log_path, 'a') as f:
                                    f.write(f"{step_count},{tcp_action}\n")
                            self.tcp_ensemble_buffer.add_action(tcp_action, step_count)

                            if self.env.enable_exp_recording and not self.use_latent_action_with_rnn_decoder:
                                self.env.get_predicted_action(tcp_action, type='full_tcp')

                        if step_count % self.gripper_action_update_interval == 0: # 현재 step이 gripper update 주기(10)에 도달했을 때 gripper action update
                            if self.use_latent_action_with_rnn_decoder:
                                gripper_action = action_all[self.gripper_latency_step:, ...]    # gripper_latency_step = 0 -> shape : (num_timesteps, D)
                            else:   # not use 
                                if action_all.shape[-1] == 4:
                                    gripper_action = action_all[self.gripper_latency_step:, 3:]
                                elif action_all.shape[-1] == 8:
                                    gripper_action = action_all[self.gripper_latency_step:, 6:]
                                elif action_all.shape[-1] == 10:
                                    gripper_action = action_all[self.gripper_latency_step:, 9:]
                                elif action_all.shape[-1] == 20:
                                    gripper_action = action_all[self.gripper_latency_step:, 18:]
                                else:
                                    raise NotImplementedError
                            # add to ensemble buffer
                            logger.debug(f"Step: {step_count}, Add gripper action to ensemble buffer: {gripper_action}")
                            self.gripper_ensemble_buffer.add_action(gripper_action, step_count)

                            if self.env.enable_exp_recording and not self.use_latent_action_with_rnn_decoder:
                                self.env.get_predicted_action(gripper_action, type='full_gripper')

                        cur_time = time.time()
                        precise_sleep(max(0., self.inference_interval_time - (cur_time - start_time))) # 주어진 inference 시간(1/inferece_fps)를 초과하지 않도록 sleep 
                        if cur_time - start_timestamp >= self.max_duration_time:
                            logger.info(f"Episode {episode_idx} reaches max duration time {self.max_duration_time} seconds.")
                            break
                        step_count += steps_per_inference # step_count = step_count + 2 
                        logger.info(f"one runner cycle time:{time.time() - start_time}")
                        # profiler.stop()
                        # profiler.print()

                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt! Terminate the episode now!")
                finally:
                    self.stop_event.set()   
                    action_thread.join()    # thread가 완전히 끝날 때까지 대기 
                    # if self.enable_video_recording:
                    #     self.stop_record_video()
                    self.env.save_exp(episode_idx)

            # TODO: support success count
            spin_thread.join()
        finally:
            self.env.destroy_node()
