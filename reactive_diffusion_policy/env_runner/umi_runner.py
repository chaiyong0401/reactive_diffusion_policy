








"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
# from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from reactive_diffusion_policy.env.real_bimanual.real_env import UmiEnv # 07/28

from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
# try:
#     from pynput.keyboard import Key, KeyCode, Listener
# except ImportError:
#     print("pynput is not available in this environment.")
#     Key, KeyCode, Listener = None, None, None
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_obs_dict_single,
                                                get_real_umi_action)
# from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.pose_util import pose_to_mat, mat_to_pose
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.policy.base_image_policy import BaseImagePolicy
OmegaConf.register_new_resolver("eval", eval, replace=True)


    
def solve_table_collision(ee_pose, gripper_width, height_threshold):    # robot의 EE와 table간 충돌 방지 height_threshold는 table의 높이로 해당 높이보다 내려가지 않도록 함 
    finger_thickness = 25.5 / 1000
    keypoints = list()  # keypoints: 그리퍼의 4꼭지점
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)  # 네 꼭지점 중 가장 낮은 z축 값보다 table 높이가 낮으면 delta 값을 이용해 그리퍼 높이 조절 
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):    # 두 robot의 EE간의 거리 측정하고 충돌 방지, threshold 초과하면 두 로봇을 이동시켜 충돌 방지 
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")    # 한번의 inference로 예측하는 action의 개수, 6개
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)

def main(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, 
    no_mirror, sim_fov, camera_intrinsics, mirror_swap):
    max_gripper_width = 0.09
    gripper_speed = 0.2
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r')) # eval_robots_config.yaml 
    
    # load left-right robot relative transform
    # bimanual setting
    # tx_left_right = np.array(robot_config_data['tx_left_right'])
    # tx_robot1_robot0 = tx_left_right
    
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']

    # load checkpoint, 'cfg'는 checkpoint에서 추출된 configuration -> model 및 dataset과 관련된 hyperparameter 

    # 수동 Hydra config 로딩
    config_dir = str(pathlib.Path(__file__).parent.joinpath('reactive_diffusion_policy', 'config'))
    config_name = "train_diffusion_unet_real_image_workspace"
    with initialize(config_path=config_dir, version_base="1.3"):  # version_base는 1.3 또는 "1.1" 등 명시
        cfg = compose(config_name=config_name)

    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    
    # cfg의 내용 확인 하기 위해 cfg_output.yaml 형태로 저장
    # with open('cfg_output2.yaml', 'w') as f:           
    #     f.write(OmegaConf.to_yaml(cfg))
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)
    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    
    with SharedMemoryManager() as shm_manager:
        # with Spacemouse(shm_manager=shm_manager) as sm, \
        #     KeystrokeCounter() as key_counter, \
            # BimanualUmiEnv(
            #     output_dir=output,
            #     robots_config=robots_config,
            #     grippers_config=grippers_config,
            #     frequency=frequency,
            #     obs_image_resolution=obs_res,
            #     obs_float32=True,
            #     camera_reorder=[int(x) for x in camera_reorder],
            #     init_joints=init_joints,
            #     enable_multi_cam_vis=True,
            #     # latency
            #     camera_obs_latency=0.17,
            #     # obs
            #     camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
            #     robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
            #     gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
            #     no_mirror=no_mirror,
            #     fisheye_converter=fisheye_converter,
            #     mirror_swap=mirror_swap,
            #     # action
            #     max_pos_speed=2.0,
            #     max_rot_speed=6.0,
            #     shm_manager=shm_manager) as env:
            
        ## One Arm Setting 
        # with Spacemouse(shm_manager=shm_manager) as sm, \
        with KeystrokeCounter() as key_counter, \
            UmiEnv(                     # franka robot set + frankainterpolationcontroller + gripper controller setting 필요
                output_dir=output,
                robot_ip='192.168.0.0',
                gripper_ip ='192.168.0.0',
                gripper_port = 1000,
                frequency=frequency,
                robot_type='franka',
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                # latency
                camera_obs_latency=0.17,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset, 일반적으로는 존재 x 
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            # cls = hydra.utils.get_class(cfg._target_)
            # print(cls)
    

            try:
                cls = hydra.utils.get_class(cfg._target_)
            except ImportError as e:
                print(f"Failed to load class {cfg._target_}: {e}")
                raise
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)


            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")
            print("Camera ready:", env.camera.is_ready) # check camera is ready
            print("Robot ready:", env.robot.is_ready)   # frankainterpolationcontroller 제어 루프가 한번 성공적으로 돌거나 모든 작업이 종료했을 때 ready
            print("Gripper ready:", env.gripper.is_ready)
            obs = env.get_obs()
            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            with torch.no_grad():
                policy.reset()
                # bimanual setting
                # obs_dict_np = get_real_umi_obs_dict(
                #     env_obs=obs, shape_meta=cfg.task.shape_meta, 
                #     obs_pose_repr=obs_pose_rep,
                #     tx_robot1_robot0=tx_robot1_robot0,
                    # episode_start_pose=episode_start_pose)
                obs_dict_np = get_real_umi_obs_dict_single(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    episode_start_pose=episode_start_pose)
                


            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                robot_states = env.get_robot_state()
                print("Robot states:", robot_states)
                for rs in robot_states:
                    print(type(rs), rs)
                
                # target_pose = np.stack([rs['TargetTCPPose'] for rs in robot_states]) # rtde controller
                target_pose = np.stack([robot_states['ActualTCPPose']]) # franka controller 

                gripper_states = env.get_gripper_state()
                print("Robot states:", gripper_states)
                for rs in gripper_states:
                    print(type(rs), rs)

                gripper_target_pos = np.asarray([gripper_states['gripper_position']])
                
                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera{match_camera}_rgb'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = (vis_img + match_img) / 2
                    obs_left_img = obs['camera0_rgb'][-1]
                    obs_right_img = obs['camera0_rgb'][-1]
                    vis_img = np.concatenate([obs_left_img, obs_right_img, vis_img], axis=1)
                    
                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            # Next episode
                            if match_episode is not None:
                                match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        elif key_stroke == KeyCode(char='w'):
                            # Prev episode
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key_stroke == KeyCode(char='m'):
                            # move the robot
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)

                            for robot_idx in range(1):
                                pos = ep[f'robot{robot_idx}_eef_pos'][0]
                                rot = ep[f'robot{robot_idx}_eef_rot_axis_angle'][0]
                                grip = ep[f'robot{robot_idx}_gripper_width'][0]
                                pose = np.concatenate([pos, rot])
                                env.robots[robot_idx].servoL(pose, duration=duration)
                                env.grippers[robot_idx].schedule_waypoint(grip, target_time=time.time() + duration)
                                target_pose[robot_idx] = pose
                                gripper_target_pos[robot_idx] = grip
                            time.sleep(duration)

                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char='a'):
                            control_robot_idx_list = list(range(target_pose.shape[0]))
                        elif key_stroke == KeyCode(char='1'):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char='2'):
                            control_robot_idx_list = [1]

                    if start_policy:
                        break

                    precise_wait(t_sample)
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay    # 현재 시각 기준 start_delay 후의 절대 시간(log, timestamp 용도)
                    t_start = time.monotonic() + start_delay    # 부팅 이후 기준 start_delay 후의 상대 시간(control loop, timer 용도)
                    env.start_episode(eval_t_start)             # start recording and return first obs 

                    # get current pose
                    obs = env.get_obs()
                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)): # robot이 한대면 robot_id = 0
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time) # 현재 시간(time.time())을 기준으로 eval_t_start - frame_latency까지 대기 
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        # 한번의 inference 
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt # t_cycle_end = t_start + {iter_idx + inference_action_horizon(6)}*dt(0.1)

                        # get obs
                        obs = env.get_obs() # x normalize 
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}') # Obs latency는 현재 시간 - observation data의 가장 최근 timestamp

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            # bimanual setting 
                            # obs_dict_np = get_real_umi_obs_dict(
                            #     env_obs=obs, shape_meta=cfg.task.shape_meta, 
                            #     obs_pose_repr=obs_pose_rep,
                            #     tx_robot1_robot0=tx_robot1_robot0,
                            #     episode_start_pose=episode_start_pose)
                            obs_dict_np = get_real_umi_obs_dict_single(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                episode_start_pose=episode_start_pose)
                            
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                            action_dict = policy.predict_action(obs_dict,
                                dataset_obs_temporal_downsample_ratio=3,
                                return_latent_action=True)
                            np_action_dict = dict_apply(action_dict,
                                lambda x: x.detach().to('cpu').numpy())
                            action_all = np_action_dict['action'].squeeze(0) # (32,29)
                            base_absolute_action = np.concatenate([ # 현재 시점의 absolute pose obs값, [-1]: 가장 최근 값 
                                                    np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                                                    np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                                                ], axis=-1)
                            action_all = np.concatenate([   # action all = (action_all의 각 timesteps base_absolute_action) -> shape(num_timesteps, latent_dim + left_tcp_pose_dim + right_tcp_pose_dim)
                                                    action_all, # shape(num_timesteps, latent_dim)
                                                    base_absolute_action[np.newaxis, :].repeat(action_all.shape[0], axis=0) # base_absolute_action을 action_all.shape[0](num_timesteps) 만큼 복제 -> shape(num_timesteps, left_tcp_pose_dim + right_tcp_pose_dim)
                                                ], axis=-1)
                            action_all = np.concatenate([  # action이 n_obs_steps 이후 timestep에 대응되므로, 이에 맞는 time index 부여 -> action all shape : [num_timestpes, latent_dim + left_tcp_pose_dim + right_tcp_pose_dim + num_timestep(1)]
                                                action_all,
                                                np.arange(self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio, action_all.shape[0] + self.n_obs_steps * self.dataset_obs_temporal_downsample_ratio)[:, np.newaxis]    #(2, num_timesteps + 2)[:,np.newaxis] -> shape (num_timesteps,1)
                                            ], axis=-1)
                            
                            tcp_step_action = tcp_step_action[:-1]  # step 제외한 나머지 -> latent 
                            gripper_step_action = gripper_step_action[:-1]
                            for step in range(steps_per_inference):
                                extended_obs = self.env.get_obs(longer_extended_obs_step,
                                                    temporal_downsample_ratio= obs_temporal_downsample_ratio)
                        
                        # convert policy action to env actions
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
                                    # 07/28 umi
                            tcp_step_action = tcp_step_action[:,:tcp_len]
                            gripper_step_action = gripper_step_action[:,tcp_len:]

                        combined_action = np.concatenate([tcp_step_action, gripper_step_action], axis=-1)

                        step_action_batch, is_bimanual = self.post_process_action(combined_action)  # shape: (T, D)
                        

                        this_target_poses = action  # 여러 time의 action 
                        assert this_target_poses.shape[1] == len(robots_config) * 7 # robot이 1대면 7, 2대면 14
                        for target_pose in this_target_poses:
                            for robot_idx in range(len(robots_config)):
                                solve_table_collision(
                                    ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                                    gripper_width=target_pose[robot_idx * 7 + 6],
                                    height_threshold=robots_config[robot_idx]['height_threshold']   #robot_config = eval_robot_config.yaml "robots"
                                )
                            
                        # deal with timing(action 실행 타이밍 조절)
                        # the same step actions are always the target for(latency로 action이 적시에 실행 안되면, 이를 보정하여 다음 가능한 timestamp에 action 스케쥴링)
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        print(dt)
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0: # 모든 action이 시간 초과
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]] # 가장 최근의 action 선택하여, 남은 action 중 마지막 action만 사용
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:   # 시간 내 실행 가능한 action만 스케줄링
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        obs_left_img = obs['camera0_rgb'][-1]
                        obs_right_img = obs['camera0_rgb'][-1]
                        vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()







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

                # tcp_step_action = tcp_step_action[-1]   # 마지막 step의 action만 사용
                # gripper_step_action = gripper_step_action[-1]
                # # logger.info(f"tcp_step_action: {tcp_step_action}")

                # tcp_step_action = tcp_step_action[:tcp_len]
                # gripper_step_action = gripper_step_action[tcp_len:]

                # 07/28 umi
                tcp_step_action = tcp_step_action[:,:tcp_len]
                gripper_step_action = gripper_step_action[:,tcp_len:]

            combined_action = np.concatenate([tcp_step_action, gripper_step_action], axis=-1)
            # convert to 16-D robot action (TCP(left,right 6D) + gripper of both arms(left, right gripper width, force))
            # TODO: handle rotation in temporal ensemble buffer!
            
            # step_action, is_bimanual = self.post_process_action(combined_action[np.newaxis, :]) # step_action : pos + rotvec
            # step_action = step_action.squeeze(0)
            # print("excute_action")
            # # logger.info(f"step action = {step_action}")
            # self.env.execute_action(step_action, use_relative_action=False, is_bimanual=is_bimanual) # env.excute_action에 use_relative_action 정의 안됨
            # # with open(tcp_pose_log_path, 'a') as f:
            # #                         f.write(f"{step_count},{np_absolute_obs_dict['left_robot_tcp_pose'][-1]}\n")
            # cur_time = time.time()
            # precise_sleep(max(0., self.control_interval_time - (cur_time - start_time)))

            # 07/28
            step_action_batch, is_bimanual = self.post_process_action(combined_action)  # shape: (T, D)
            action_timestamps = (np.arange(len(step_action_batch), dtype=np.float64)) * self.dt + self.obs_timestamps[-1]
            print(self.dt)
            action_exec_latency = 0.01
            curr_time = time.time()
            is_new = action_timestamps > (curr_time + action_exec_latency)
            if np.sum(is_new) == 0: # 모든 action이 시간 초과
                # exceeded time budget, still do something
                this_target_poses = combined_action[[-1]] # 가장 최근의 action 선택하여, 남은 action 중 마지막 action만 사용
                # schedule on next available step
                next_step_idx = int(np.ceil((curr_time - self.eval_t_start) / self.dt))
                action_timestamp = self.eval_t_start + (next_step_idx) * self.dt
                print('Over budget', action_timestamp - curr_time)
                action_timestamps = np.array([action_timestamp])
            else:   # 시간 내 실행 가능한 action만 스케줄링
                this_target_poses = this_target_poses[is_new]
                action_timestamps = action_timestamps[is_new]

            self.env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
            print(f"Submitted {len(this_target_poses)} steps of actions.")
            self.action_step_count += 1