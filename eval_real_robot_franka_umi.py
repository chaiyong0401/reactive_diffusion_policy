# # 07/29
# import pathlib
# import torch
# import dill
# import hydra
# from omegaconf import OmegaConf
# from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
# from reactive_diffusion_policy.policy.base_image_policy import BaseImagePolicy

# import os
# import psutil
# from loguru import logger

# os.environ["OPENBLAS_NUM_THREADS"] = "12"
# os.environ["MKL_NUM_THREADS"] = "12"
# os.environ["NUMEXPR_NUM_THREADS"] = "12"
# os.environ["OMP_NUM_THREADS"] = "12"

# OmegaConf.register_new_resolver("eval", eval, replace=True)
# from omegaconf import DictConfig
# torch.serialization.add_safe_globals({'omegaconf.dictconfig.DictConfig': DictConfig})

# @hydra.main(
#     version_base=None,
#     config_path=str(pathlib.Path(__file__).parent.joinpath(
#         'reactive_diffusion_policy', 'config')),
#     config_name="train_diffusion_unet_real_image_workspace"
# )
# def main(cfg):
#     ckpt_path = cfg.ckpt_path
#     payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, weights_only=False) # 07/30
#     logger.debug(f"Loaded checkpoint from {ckpt_path}")
#     cls = hydra.utils.get_class(cfg._target_)
#     workspace = cls(cfg)
#     workspace: BaseWorkspace
#     workspace.load_payload(payload, exclude_keys=None, include_keys=None)
#     logger.debug(f"Loaded workspace from {ckpt_path}")
#     if 'diffusion' in cfg.name:
#         policy: BaseImagePolicy = workspace.model
#         if cfg.training.use_ema:
#             policy = workspace.ema_model
#         if 'latent' in cfg.name:
#             policy.at.set_normalizer(policy.normalizer)
#         device = torch.device('cuda')
#         policy.eval().to(device)
#         policy.num_inference_steps = 8
#         policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
#     else:
#         raise NotImplementedError
#     logger.debug(f"Policy is ready for evaluation")
#     env_runner = hydra.utils.instantiate(cfg.task.env_runner)
#     env_runner.run(policy)
#     logger.debug("Environment runner has started evaluation")
# if __name__ == '__main__':
#     main()


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
from hydra import initialize, compose
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
                                                get_real_umi_action,
                                                get_real_obs_dict_rdp)  # 07/31
# from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.pose_util import pose_to_mat, mat_to_pose
from reactive_diffusion_policy.workspace.base_workspace import BaseWorkspace
from reactive_diffusion_policy.policy.base_image_policy import BaseImagePolicy
OmegaConf.register_new_resolver("eval", eval, replace=True)
from loguru import logger
import threading, queue
from collections import deque
import numpy as np
np.set_printoptions(precision=15, suppress=True)

from reactive_diffusion_policy.common.action_utils import (
    interpolate_actions_with_ratio,
    relative_actions_to_absolute_actions,
    absolute_actions_to_relative_actions,
    get_inter_gripper_actions
)
from reactive_diffusion_policy.common.space_utils import ortho6d_to_rotation_matrix, pose10d_to_mat, mat_to_pose

# Attention 가중치를 저장할 글로벌 변수 #09/09
attention_maps = list()

# Hook 함수 정의
def get_attention_map(name):
    def hook(model, input, output):
        # output[0]는 어텐션 가중치 텐서일 수 있습니다. 모델 구조에 따라 다름
        # 보통 (batch_size, num_heads, sequence_length, sequence_length) 형태
        attention_maps.append(output.detach().cpu())
    return hook

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


def at_worker(env, policy, latent_q, stop_event,
              temporal_downsample_ratio : int, control_dt: float, shape_meta):
    """
    policy.at 를 이용해 latent → real action, 그리고 env.exec_actions 실행.
    control_dt 간격으로 sleep 후 실행, stop_event가 set되면 종료.
    """
    device = torch.device('cuda')                 # or 'cpu'
    # at = policy.at.eval().to(device)
    action_dim = 9
    dt = control_dt
    # if at is None:
    #     raise RuntimeError("policy.at is None at worker start")
    next_call = time.time()
    action_horizon = 12

    while not stop_event.is_set():
        try:
            latent = latent_q.get(timeout=0.02)   # (D_latent + meta)
            # logger.debug(f"Received latent with shape: {latent.shape}")
        except queue.Empty:
            # logger.debug("No latent received, continuing...")
            time.sleep(max(0,next_call-time.time()))
            next_call += dt
            continue
        

        # ---- 1) 메타 분리 ----
        timestamps  = float(latent[-1])               # 마지막 컬럼에 time-step
        latent = latent[:-1]                  # 마지막 컬럼 제거 (step)
        step_count = int(latent[-1])          # 마지막 컬럼에 step count
        latent = latent[:-1]                  # 마지막 컬럼 제거 (step)
        abs_pose  = latent[-action_dim:]      # base absolute TCP pose
        latent_z  = latent[:-action_dim]
        # logger.debug(f"step_idx: {step_count}, timestamps: {timestamps} abs_pose: {abs_pose}, latent_z shape: {latent_z.shape}")

        # ---- 2) extended obs 가져오기 ----
        # ext_obs = env.get_obs()
        ext_obs = env.get_extend_obs()    # 08/22
        obs_timestamps = ext_obs['timestamp']
        obs_timestamps = obs_timestamps[-1]
        ext_obs_np = get_real_obs_dict_rdp(ext_obs,
                            shape_meta=shape_meta, is_extended_obs=True)
        
        # for k in list(ext_obs_np.keys()):   # 3dtacdex3d
        #     if k.endswith('tcp_wrench') or k == 'tcp_wrench':
        #         # 학습과 동일하게 합력(3) 사용
        #         logger.info(f"Processing {k} with shape: {ext_obs_np[k].shape}")   # (2,48)
        #         logger.debug(f"tcp_wrench shape before processing: {ext_obs_np[k]}")  # (2,48)
        #         logger.info(f"Processed {k} to shape: {ext_obs_np[k].shape}")  # (2,3)
        #         logger.debug(f"tcp_wrench shape after processing: {ext_obs_np[k]}")  # (2,3)

        ext_obs_t = dict_apply(ext_obs_np,
                               lambda x: torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device))

        # ---- 3) latent → δTCP ----

        # dataset_obs_temporal_downsample_ratio = 2
        # tcp_step_action = policy.predict_from_latent_action(tcp_step_latent_action, extended_obs_dict, tcp_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
        # gripper_step_action = policy.predict_from_latent_action(gripper_step_latent_action, extended_obs_dict, gripper_extended_obs_step, dataset_obs_temporal_downsample_ratio)['action'][0].detach().cpu().numpy()
        tcp_base_absolute_action = abs_pose
        gripper_base_absolute_action = abs_pose
        tcp_step_latent_action =  torch.from_numpy(latent_z.astype(np.float32)).unsqueeze(0)   # [1,32]
        # tcp_step_latent_action_repeated = tcp_step_latent_action.repeat(29, 1)
        gripper_step_latent_action = torch.from_numpy(latent_z.astype(np.float32)).unsqueeze(0)  
        # logger.debug(f"tcp_step_latent_action shape: {tcp_step_latent_action.shape}, tcp_step_action: {tcp_step_latent_action}")
        with torch.no_grad():
            delta = policy.predict_from_latent_action(
                        tcp_step_latent_action,
                        ext_obs_t,
                        step_count,
                        # temporal_downsample_ratio)['action'][0].cpu().numpy()
                        temporal_downsample_ratio,
                        extend_obs_pad_after=True)['action'][0].cpu().numpy()

        # logger.debug(f"delta shape: {delta.shape}") # (27,10)

        # delta_tcp = delta[:,:action_dim]  # TCP pose delta
        # delta_gripper = delta[:,action_dim:]  # gripper width delta
        delta_horizon = delta[:action_horizon, :]  # TCP pose delta
        # r = 3
        # delta_horizon = delta[::r, :]
        # logger.debug(f"delta_horizon shape: {delta_horizon.shape}") 
        abs_action = relative_actions_to_absolute_actions(delta_horizon, abs_pose)

        # logger.debug(f"abs_action shape: {abs_action.shape}")   # (action horizon,10)
        tcp_abs_action = abs_action[:,:action_dim]
        gripper_abs_action = abs_action[:,action_dim:]
        left_action_batch = pose10d_to_mat(tcp_abs_action)
        left_action_6d = mat_to_pose(left_action_batch) # (pos + rotvec)

        # ---- 4) 충돌 보정 & post process ----
        combined_action = np.concatenate([left_action_6d,gripper_abs_action],axis=1)    #  6D pose + 1D gripper width
        # logger.debug(f"combined_action shape: {combined_action.shape}, combined_action: {combined_action}") #(9,7)
        action_timestamps = timestamps + np.arange(len(combined_action), dtype=np.float64) * dt
        # logger.debug(f"action_timestamps: {action_timestamps}")
        action_timestamps = obs_timestamps + np.arange(len(combined_action), dtype=np.float64) * dt
        # base = env.next_robot_time(margin=0.02)   # or use env.make_future_times
        # action_timestamps = base + np.arange(len(combined_action), dtype=np.float64) * dt
        # logger.debug(f"action_timestamps with scheduled_waypoint_timestamps: {action_timestamps}")


        action_exec_latency = 0.01
        curr_time = time.time()
        is_new = action_timestamps > (curr_time + action_exec_latency)
        # logger.debug(f"len of combined_action: {len(combined_action)}") # action horizon
        # logger.debug(f"len of action_timestamps: {len(action_timestamps)}") # action horizon
        # timestamp = np.array([time.time()+control_dt])

        if np.sum(is_new) == 0:
            combined_action = combined_action[[-1]]
            # logger.debug(f"No new actions, using last action: {combined_action}")
        else:
            combined_action = combined_action[is_new]
            action_timestamps = action_timestamps[is_new]
            # logger.info(f"New actions: {len(combined_action)}")
        # ---- 5) 실제 로봇에 전송 ----
        # only 2 step shoot 08/13
        # combined_action = combined_action[:3]
        # action_timestamps = action_timestamps[:3]
        # original action
        env.rdp_exec_actions(combined_action, action_timestamps, compensate_latency=True)
        # env.rdp_exec_actions(combined_action, action_timestamps, compensate_latency=False)
        # logger.debug(f"submit {len(combined_action)}")

        # combined_last_action = combined_action[-1]  # 마지막 액션만 전송
        # last_action_timestamps = action_timestamps[-1]  # 마지막 액션의 타임스탬프
        # env.rdp_exec_actions(combined_last_action, last_action_timestamps,compensate_latency=True)

        now=time.time()
        sleep_time = max(0, next_call - now)   
        time.sleep(sleep_time)
        next_call += dt




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
@click.option('--steps_per_inference', '-si', default=4, type=int, help="Action horizon for inference.")    # 한번의 inference로 예측하는 action의 개수, 6개
# @click.option('--steps_per_inference', '-si', default=12, type=int, help="Action horizon for inference.")    # 한번의 inference로 예측하는 action의 개수, 6개
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
# @click.option('--frequency', '-f', default=12, type=float, help="Control frequency in Hz.")
@click.option('--frequency', '-f', default=24, type=float, help="Control frequency in Hz.")
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
    # latent_q   = queue.Queue(128)   # 07/31
    # latent_q = deque(maxlen=256) 
    stop_event = threading.Event()  # 07/31
    
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
    # config_dir = str(pathlib.Path(__file__).parent.joinpath('reactive_diffusion_policy', 'config'))
    os.chdir(str(pathlib.Path(__file__).parent))  # Hydra가 상대 경로 기준으로 삼을 디렉토리로 이동

    config_dir = "reactive_diffusion_policy/config"  # 상대 경로로 변경
    config_name = "train_latent_diffusion_unet_real_image_workspace"
    # with initialize(config_path=config_dir, version_base="1.3"):  # version_base는 1.3 또는 "1.1" 등 명시
    #     cfg = compose(config_name=config_name)

    with initialize(config_path=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_name,
                  overrides=[
                      "task=real_umi_image_xela_wrench_ldp_24fps",
                      "at=at_peel",
                      "+task.env_runner.output_dir=/home/embodied-ai/mcy/reactive_diffusion_policy_umi/saving_video",
                      '+ckpt_path="/home/embodied-ai/mcy/reactive_diffusion_policy_umi/data/outputs/xela_peg_hole/2025.08.28/00.10.59_train_latent_diffusion_unet_image_real_umi_image_xela_wrench_ldp_24fps_0828001057/checkpoints/latest.ckpt"',
                      'at_load_dir="/home/embodied-ai/mcy/reactive_diffusion_policy_umi/data/outputs/xela_peg_hole/2025.08.27/17.05.27_train_vae_real_umi_image_xela_wrench_at_24fps_0827170527/checkpoints/latest.ckpt"',
                  ])

    A = input
    # if not ckpt_path.endswith('.ckpt'):
    #     ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    # payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill, weights_only=False) # 07/30
    # cfg = payload['cfg']
    
    
    # cfg의 내용 확인 하기 위해 cfg_output.yaml 형태로 저장
    # with open('cfg_output2.yaml', 'w') as f:           
    #     f.write(OmegaConf.to_yaml(cfg))
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)
    # setup experiment
    dt = 1/frequency

    # # ==== 경로 세팅 ====
    # TACTILE_CKPT = "/home/embodied-ai/mcy/universal_manipulation_interface/tacdex3d/pretraining/logging_data/pretrain_four_train:tactile_play_data_train_test:tactile_play_data_test_rpr_0.0__mp_100000_wd_0_gat_gat_0/checkpoint_9000.pt"
    # TACTILE_STATS = "/home/embodied-ai/mcy/universal_manipulation_interface/tacdex3d/output/xela_data_force_stats.npy"

    # # CPU 권장(지연 적고 안정적), GPU 쓰려면 device="cuda"
    # TACTILE_ENCODER = TactileGraphEncoder(ckpt_path=TACTILE_CKPT, stats_path=TACTILE_STATS, device="cpu")


    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=(224, 224),  # 07/30
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    
    with SharedMemoryManager() as shm_manager:
            
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
                obs_image_resolution=(224,224), # 07/30
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                # latency
                camera_obs_latency=0.17,
                # obs
                camera_obs_horizon=2,   # 07/30
                robot_obs_horizon=2,    # 07/30
                gripper_obs_horizon=2,  # 07/30
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
            n_obs_steps =  2   # 07/30
            temporal_downsample_ratio = 2
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
    
            ckpt_path = cfg.ckpt_path
            payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, weights_only=False) # 07/30
            logger.debug(f"Loaded checkpoint from {ckpt_path}")

            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            logger.debug(f"Loaded workspace from {ckpt_path}")
            if 'diffusion' in cfg.name:
                policy: BaseImagePolicy = workspace.model
                if cfg.training.use_ema:
                    policy = workspace.ema_model
                if 'latent' in cfg.name:
                    policy.at.set_normalizer(policy.normalizer)
                device = torch.device('cuda')
                policy.eval().to(device)
                policy.num_inference_steps = 8
                policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
            else:
                raise NotImplementedError

            logger.info(f"cfg.name  :{cfg.name}")
            logger.info(f"model type: {type(policy)}")

            # policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = 'relative'
            action_pose_repr = 'relative'
            # print('obs_pose_rep', obs_pose_rep)
            # print('action_pose_repr', action_pose_repr)


            device = torch.device('cuda')
            policy.eval().to(device)
            # attention map 시각화 09/09
            print("Attributes of policy.obs_encoder:")
            print(dir(policy.obs_encoder))
            try:
                # ViT 모델의 마지막 어텐션 레이어에 접근 (이 경로는 모델에 따라 다를 수 있음)
                left_camera_model = policy.obs_encoder.key_model_map['camera0']
                # last_attention_layer = policy.obs_encoder.model.blocks[-1].attn.attn_drop
                last_attention_layer = left_camera_model.blocks[-1].attn.attn_drop
                last_attention_layer.register_forward_hook(get_attention_map("last_attention"))
                print("Successfully registered hook to the last attention layer of camera0.")
            except AttributeError as e:
                print(f"Could not register hook. Model structure might be different: {e}")
            
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
                    # _ = cv2.pollKey()
                    cv2.waitKey(1)
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
                
                # ========== policy control loop(inference latent) ==============
                try:
                    # start episode
                    policy.reset()
                    latent_q = queue.Queue(256)   # 07/31
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay    # 현재 시각 기준 start_delay 후의 절대 시간(log, timestamp 용도)
                    t_start = time.monotonic() + start_delay    # 부팅 이후 기준 start_delay 후의 상대 시간(control loop, timer 용도)
                    env.start_episode(eval_t_start)             # start recording and return first obs 
                    ##########################################
                    stop_event.clear()
                    time.sleep(0.5)
                    at_thread = threading.Thread(   # 07/30
                    target=at_worker,
                    args=(env, policy, latent_q, stop_event, temporal_downsample_ratio, 1/frequency, cfg.task.shape_meta),
                    daemon=True)
                    at_thread.start()
                    ##########################################
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
                        while not latent_q.empty():
                            try:
                                latent_q.get_nowait()   # 항목 하나 꺼내기
                            except queue.Empty:
                                break
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt # t_cycle_end = t_start + {iter_idx + inference_action_horizon(6)}*dt(0.1)

                        # get obs
                        obs = env.get_obs() # x normalize 
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}') # Obs latency는 현재 시간 - observation data의 가장 최근 timestamp
                        # logger.info(f"rdp inference obs_timestamps: {obs_timestamps}")
                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            # bimanual setting 
                            # obs_dict_np = get_real_umi_obs_dict(
                            #     env_obs=obs, shape_meta=cfg.task.shape_meta, 
                            #     obs_pose_repr=obs_pose_rep,
                            #     tx_robot1_robot0=tx_robot1_robot0,
                            #     episode_start_pose=episode_start_pose)
                            obs_dict_np, base_pose = get_real_umi_obs_dict_single( # relative obs 
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                episode_start_pose=episode_start_pose)
                            
                            # # === 추가: tcp_wrench 임베딩 치환 ===
                            # # 실제 key 이름이 'tcp_wrench'가 아닐 수 있으니 끝이 'tcp_wrench'인 항목을 모두 변환
                            # for k in list(obs_dict_np.keys()):
                            #     if k.endswith('tcp_wrench') or k == 'tcp_wrench':
                            #         obs_dict_np[k] = TACTILE_ENCODER.encode(obs_dict_np[k])  # (T,256)
                            
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                            action_dict = policy.predict_action(obs_dict,
                                temporal_downsample_ratio,
                                return_latent_action=True)
                            np_action_dict = dict_apply(action_dict,
                                lambda x: x.detach().to('cpu').numpy())
                            action_all = np_action_dict['action'].squeeze(0) # (32,29)
                            base_absolute_action = base_pose
                            # base_absolute_action = np.concatenate([ # 현재 시점의 absolute pose obs값, [-1]: 가장 최근 값 
                            #                         np_absolute_obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in np_absolute_obs_dict else np.array([]),
                            #                         np_absolute_obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in np_absolute_obs_dict else np.array([])
                            #                     ], axis=-1)
                            action_all = np.concatenate([   # action all = (action_all의 각 timesteps base_absolute_action) -> shape(num_timesteps, latent_dim + left_tcp_pose_dim + right_tcp_pose_dim)
                                                    action_all, # shape(num_timesteps, latent_dim)
                                                    base_absolute_action[np.newaxis, :].repeat(action_all.shape[0], axis=0) # base_absolute_action을 action_all.shape[0](num_timesteps) 만큼 복제 -> shape(num_timesteps, left_tcp_pose_dim + right_tcp_pose_dim)
                                                ], axis=-1)
                            # logger.info(f"action all shape: {action_all.shape}, action all: {action_all}")
                            action_all = np.concatenate([  # action이 n_obs_steps 이후 timestep에 대응되므로, 이에 맞는 time index 부여 -> action all shape : [num_timestpes, latent_dim + left_tcp_pose_dim + right_tcp_pose_dim + num_timestep(1)]
                                                action_all,
                                                np.arange(policy.n_obs_steps * temporal_downsample_ratio, action_all.shape[0] + policy.n_obs_steps * temporal_downsample_ratio)[:, np.newaxis]    #(2, num_timesteps + 2)[:,np.newaxis] -> shape (num_timesteps,1)
                                            ], axis=-1) # 07/30
                            # logger.info(f"action all shape after time index: {action_all.shape}, action all: {action_all}")
                            # action_timestamps = (np.arange(len(action_all), dtype=np.float64)
                            action_timestamps = (np.arange(action_all.shape[0], dtype=np.float64)
                                ) * dt + obs_timestamps[-1]
                            # for k, z in enumerate(action_all):
                            #     z_full = np.concatenate([z, [action_timestamps[k]]], axis=0)
                            #     latent_q.put(z_full)
                            # logger.debug(f"action_all shape: {action_all.shape}, action_all len: {len(action_all)}")
                            for k, z in enumerate(action_all):
                                # logger.info(f"rdp inference action {k}: {z}")
                                z_full = np.concatenate([z, [action_timestamps[k]]], axis=0)
                                latent_q.put(z_full)

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

                        # attention map 시각화 09/09
                        if attention_maps:
                            # 가장 최근의 attention map 가져오기
                            att_mat = torch.stack(attention_maps).squeeze(0)
                            attention_maps.clear() # 리스트 비우기

                            # CLS 토큰을 제외하고, 평균 어텐션 가중치 계산
                            att_mat = att_mat.mean(dim=1)[:, 1:, 1:] # 헤드 평균, CLS 토큰 제외

                            # 이미지 크기에 맞게 리사이즈
                            h, w = obs_left_img.shape[:2]
                            # ViT는 이미지를 패치 단위로 처리하므로 패치 수에 맞는 그리드 크기를 계산해야 합니다.
                            num_patches = att_mat.shape[-1]
                            grid_size = int(num_patches**0.5)

                            att_mat = att_mat.reshape(-1, grid_size, grid_size)
                            att_mat = F.interpolate(att_mat.unsqueeze(0), scale_factor=(h/grid_size, w/grid_size), mode='bilinear', align_corners=False).squeeze(0).cpu().numpy()

                            # 시각화를 위해 Normalize
                            att_mat = (att_mat - att_mat.min()) / (att_mat.max() - att_mat.min())

                            # Colormap 적용
                            heatmap = cv2.applyColorMap(np.uint8(255 * att_mat[0]), cv2.COLORMAP_JET)
                            heatmap = np.float32(heatmap) / 255

                            # 원본 이미지와 합치기
                            vis_img_with_att = (heatmap * 0.5 + np.float32(obs_left_img) * 0.5)

                            cv2.imshow('Attention Map', vis_img_with_att)
                        cv2.imshow('default', vis_img[...,::-1])

                        # _ = cv2.pollKey()
                        cv2.waitKey(1)
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                # env._harvest_planned_samples()
                                # env.plot_trajectory(source='waypoint', smooth=True, dt=0.01, block=False)  # 스케줄한 목표 경로
                                # env.plot_trajectory(source='actual',   smooth=True, dt=0.01, block=False)
                                # env.plot_waypoint_smoothed(dt=0.01, block=False)
                                # env.plot_waypoints_colored(max_legend_groups=16, legend_outside=True)
                                # env.plot_waypoints_colored_with_actual(kind="robot", smooth=True, dt=0.01)
                                env.plot_waypoints_vs_actual_aligned(kind="robot", dt=0.01, block=True)

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
                        logger.info(f"rdp inference iter_idx: {iter_idx} ")

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()

                finally:    # 07/30
                    stop_event.set()
                    at_thread.join()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()

