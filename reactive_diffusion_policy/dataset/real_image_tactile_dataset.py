from typing import Dict
import torch
import numpy as np
import os
from threadpoolctl import threadpool_limits
import copy
import tqdm
from reactive_diffusion_policy.common.pytorch_util import dict_apply
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from reactive_diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from reactive_diffusion_policy.common.normalize_util import (
    get_image_range_normalizer,
    get_action_normalizer,
    get_range_normalizer_from_stat,
    array_to_stats,
    get_identity_normalizer_from_stat
)
from reactive_diffusion_policy.common.action_utils import absolute_actions_to_relative_actions, get_inter_gripper_actions
from reactive_diffusion_policy.real_world.real_world_transforms import RealWorldTransforms
import time
import torch.utils.data
from loguru import logger

# 08/07 3dtacdex3d
from tacdex3d.pretraining.models.edcoder import PreModel
from tacdex3d.robomimic.models.utils import data_to_gnn_batch
# from tacdex3d.robomimic.models.base_nets import MAEGAT
from tacdex3d.diffusion_policy.real_world.fk.constants import (
    XELA_USPA44_COORD,
    XELA_TACTILE_ORI_COORD,
)
from scipy.spatial.transform import Rotation as R

class RealImageTactileDataset(BaseImageDataset):
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: str,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 n_obs_steps=None,
                 obs_temporal_downsample_ratio=1, # for latent diffusion
                 n_latency_steps=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 delta_action=False,
                 relative_action=False,
                 relative_tcp_obs_for_relative_action=True,
                 transform_params=None,
                 use_constant_rgb: bool = False,    # 07/07
                 constant_rgb_value: float = 0.5,   # 07/07
                 ):
        print("realimgaetactiledataset")
        print(dataset_path)
        # assert os.path.isdir(dataset_path)

        print("realimgaet")
        # 08/07
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
        # soft gripper
        # ckpt_path = "/home/embodied-ai/mcy/universal_manipulation_interface/tacdex3d/pretraining/logging_data/pretrain_four_train:tactile_play_data_train_test:tactile_play_data_test_rpr_0.0__mp_100000_wd_0_gat_gat_0/checkpoint_9000.pt"
        # hard gripper
        ckpt_path = "/home/embodied-ai/mcy/universal_manipulation_interface/tacdex3d/pretraining/logging_data/pretrain_four_train:tactile_play_data_train_test:tactile_play_data_test_rpr_0.0__mp_100000_wd_0_gat_gat_62/checkpoint_8500.pt"
        
        state_dict =torch.load(ckpt_path, map_location='cpu')
        self.encoder.nets.load_state_dict(state_dict)
        for param in self.encoder.parameters():
            param.requires_grad = False
        # stats = np.load("/home/embodied-ai/mcy/universal_manipulation_interface/tacdex3d/output/xela_stack_data_force_stats.npy", allow_pickle=True).item() # stack
        stats = np.load("/home/embodied-ai/mcy/universal_manipulation_interface/tacdex3d/output/xela_data_force_stats.npy", allow_pickle=True).item()
        self.mean = stats["mean"]  # shape: (3,)
        self.std = stats["std"]

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
                print("key:", key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
                print("low_dim:", key)

        extended_rgb_keys = list()
        extended_lowdim_keys = list()
        extended_obs_shape_meta = shape_meta.get('extended_obs', dict())
        for key, attr in extended_obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                extended_rgb_keys.append(key)
            elif type == 'low_dim':
                extended_lowdim_keys.append(key)

        # zarr_path = os.path.join(dataset_path, 'replay_buffer.zarr')
        zarr_path = os.path.join(dataset_path)
        zarr_load_keys = set(rgb_keys + lowdim_keys + extended_rgb_keys + extended_lowdim_keys + ['action']+ ['left_robot_tcp_pos'] + ['left_robot_rot_axis_angle'])
        zarr_load_keys = list(filter(lambda key: "wrt" not in key, zarr_load_keys))
        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=zarr_load_keys)

        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff
        
        self.relative_action = relative_action
        self.relative_tcp_obs_for_relative_action = relative_tcp_obs_for_relative_action
        self.transforms = RealWorldTransforms(option=transform_params)
        self.use_constant_rgb = use_constant_rgb
        self.constant_rgb_value = constant_rgb_value

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                if key not in extended_rgb_keys + extended_lowdim_keys:
                    key_first_k[key] = n_obs_steps * obs_temporal_downsample_ratio
        self.key_first_k = key_first_k

        self.seed = seed
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        print("=== [DEBUG] ===")
        print("train mask sum (should be >0):", train_mask.sum())
        print("n_episodes:", replay_buffer.n_episodes)
        print("episode ends:", replay_buffer.episode_ends)
        print("================")
        print("Available keys:", replay_buffer.keys())
        # obs_path_init = f"/home/embodied-ai/mcy/reactive_diffusion_policy_umi/dataset_obs_log_init.txt"  # ← 원하는 경로로 변경 가능 # 07/13
        # with open(obs_path_init, "a") as f:  # append 모드
        #     f.write(f"tcp_pose : {replay_buffer['left_robot_tcp_pos'][:, :3]}\n")

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.extended_rgb_keys = extended_rgb_keys
        self.extended_lowdim_keys = extended_lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.obs_downsample_ratio = obs_temporal_downsample_ratio
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # calculate relative action / obs
        # not use
        if "left_robot_wrt_right_robot_tcp_pose" in self.lowdim_keys or "right_robot_wrt_left_robot_tcp_pose" in self.lowdim_keys:
            inter_gripper_data_dict = {key: list() for key in self.lowdim_keys if 'robot_tcp_pose' in key and 'wrt' in key}
            for data in tqdm.tqdm(self, leave=False, desc='Calculating inter-gripper relative obs for normalizer'):
                for key in inter_gripper_data_dict.keys():
                    inter_gripper_data_dict[key].append(data['obs'][key])
            inter_gripper_data_dict = dict_apply(inter_gripper_data_dict, np.stack)

        # use
        if self.relative_action:
            print(self.lowdim_keys)
            keys_to_collect = [key for key in (self.lowdim_keys + ['action']) if ('robot_tcp_pose' in key and 'wrt' not in key) or 'action' in key]
            print("✅ relative_data_dict keys to collect:", keys_to_collect) # left_robot_tcp_pose, action 
            print("Dataset length:", len(self)) # 25000
            if len(self) == 0:
                raise RuntimeError("❗ Dataset is empty. No samples to process for normalizer calculation.")

            relative_data_dict = {key: list() for key in (self.lowdim_keys + ['action']) if ('robot_tcp_pose' in key and 'wrt' not in key) or 'action' in key}

            print("=== [DEBUG] Sampler length ===")
            print(len(self.sampler))

            for idx in range(len(self.sampler)):
                sample = self.sampler.sample_sequence(idx)
                print(f"Sample {idx} ok")
                break
            sample0 = self.sampler.sample_sequence(0)
            print("DEBUG: sample type: ", type(sample0))
            if isinstance(sample0, dict):
                print("DEBUG: sample0 keys:", list(sample0.keys()))
                for k, v in sample0.items():
                    if hasattr(v, "shape"):
                        print(f"  - {k}: array, shape {v.shape}")
                    elif isinstance(v, (list, tuple)):
                        print(f"  - {k}: list/tuple, len {len(v)}")
                    else:
                        print(f"  - {k}: {type(v)}, value {v}")

            ######################################################
            for data in tqdm.tqdm(self, leave=False, desc='Calculating relative action/obs for normalizer'):
            # for idx in tqdm.tqdm(range(len(self.sampler)), desc='Calculating relative action/obs for normalizer'):
            #     data = self.sampler.sample_sequence(idx)

                for key in relative_data_dict.keys():
                    if key == 'action':
                        
                        relative_data_dict[key].append(data[key])
                        # print("action:", data[key])
                    else:
                        relative_data_dict[key].append(data['obs'][key])
                        # print("left_robot_tcp_pose:", data['obs'][key])
                        
                        # relative_data_dict[key].append(data[key])
                        # print("left_robot_tcp_pose:", data[key])
            
            # data loading check
            for key, arr in relative_data_dict.items():
                print(f"DEBUG: '{key}' has {len(arr)} samples") # left_robot_tcp_pose, action 

            relative_data_dict = dict_apply(relative_data_dict, np.stack) # 25000 tcp_pose, action data dict

        # action
        if self.relative_action:
            action_all = relative_data_dict['action']
        else:
            action_all = self.replay_buffer['action'][:, :self.shape_meta['action']['shape'][0]]

        normalizer['action'] = get_action_normalizer(action_all)

        # obs
        # obs_path = f"/home/embodied-ai/mcy/reactive_diffusion_policy_umi/dataset_obs_log.txt"  # ← 원하는 경로로 변경 가능 # 07/13
        # with open(obs_path, "a") as f:  # append 모드
        for key in list(set(self.lowdim_keys)):
            # logger.debug(f"lowdim_kys chekc##################")
            if self.relative_action and key in relative_data_dict:
                normalizer[key] = get_action_normalizer(relative_data_dict[key])
                # f.write(f"robot_tcp_pose: {relative_data_dict[key]}\n")
            elif 'robot_tcp_pose' in key and 'wrt' in key:
                normalizer[key] = get_action_normalizer(inter_gripper_data_dict[key])
            elif 'robot_tcp_pose' in key and 'wrt' not in key:
                normalizer[key] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])
            # elif 'robot_tcp_wrench' in key:  # ✅ 여기에 추가
            #     normalizer[key] = get_action_normalizer(
            #         self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])
            elif 'robot_tcp_wrench' in key:  # 7/29
                ########## original code ##########
                # stat = array_to_stats(self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])
                # normalizer[key] = get_range_normalizer_from_stat(stat)
                # # print(f"[DEBUG] key: {key}")
                # # print("[DEBUG] tactile offset scale: ", normalizer[key].params_dict['scale'])
                # # print("[DEBUG] tactile offset offset: ", normalizer[key].params_dict['offset'])
                # # print("Input Stats (min):", normalizer[key].params_dict['input_stats']['min'])
                # # print("Input Stats (max):", normalizer[key].params_dict['input_stats']['max'])
                # # print("Input Stats (mean):", normalizer[key].params_dict['input_stats']['mean'])
                # # print("Input Stats (std):", normalizer[key].params_dict['input_stats']['std']) 

                ########### 08/07 3dtacdex3d
                stat = array_to_stats(self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])
                normalizer[key] = get_identity_normalizer_from_stat(stat)
                print(f"[DEBUG] key: {key}")
                print("[DEBUG] tactile offset scale: ", normalizer[key].params_dict['scale'])
                print("[DEBUG] tactile offset offset: ", normalizer[key].params_dict['offset'])
                print("Input Stats (min):", normalizer[key].params_dict['input_stats']['min'])
                print("Input Stats (max):", normalizer[key].params_dict['input_stats']['max'])
                print("Input Stats (mean):", normalizer[key].params_dict['input_stats']['mean'])
                print("Input Stats (std):", normalizer[key].params_dict['input_stats']['std']) 
                logger.info(f"[DEBUG] key: {key} 3dtacdex no normalizer")
            else:
                normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                    self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])

        for key in list(set(self.extended_lowdim_keys)): # no duplicate
            if key in self.lowdim_keys:
                assert self.shape_meta['extended_obs'][key]['shape'][0] == self.shape_meta['obs'][key]['shape'][0], \
                    f"Extended obs {key} has different shape from obs {key}"
            else:
                if self.relative_action and key in relative_data_dict:
                    normalizer[key] = get_action_normalizer(relative_data_dict[key])
                elif 'robot_tcp_pose' in key and 'wrt' in key:
                    normalizer[key] = get_action_normalizer(inter_gripper_data_dict[key])
                elif 'robot_tcp_pose' in key and 'wrt' not in key: # not used now
                    normalizer[key] = get_action_normalizer(self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])
                # elif 'robot_tcp_wrench' in key:  # ✅ 여기에 추가
                #     normalizer[key] = get_action_normalizer(
                #         self.replay_buffer[key][:, :self.shape_meta['obs'][key]['shape'][0]])
                elif 'robot_tcp_wrench' in key:  # 7/29
                    ################### original code ###################
                    # stat = array_to_stats(self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])
                    # normalizer[key] = get_range_normalizer_from_stat(stat)
                    # # print(f"[DEBUG] key: {key}")
                    # # print("[DEBUG] tactile offset scale: ", normalizer[key].params_dict['scale'])
                    # # print("[DEBUG] tactile offset offset: ", normalizer[key].params_dict['offset'])
                    # # print("Input Stats (min):", normalizer[key].params_dict['input_stats']['min'])
                    # # print("Input Stats (max):", normalizer[key].params_dict['input_stats']['max'])
                    # # print("Input Stats (mean):", normalizer[key].params_dict['input_stats']['mean'])
                    # # print("Input Stats (std):", normalizer[key].params_dict['input_stats']['std'])
                    # ################ 08/07 3dtacdex3d
                    stat = array_to_stats(self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])
                    normalizer[key] = get_identity_normalizer_from_stat(stat)
                    # # print(f"[DEBUG] key: {key}")
                    print("[DEBUG] tactile offset scale: ", normalizer[key].params_dict['scale'])
                    print("[DEBUG] tactile offset offset: ", normalizer[key].params_dict['offset'])
                    print("Input Stats (min):", normalizer[key].params_dict['input_stats']['min'])
                    print("Input Stats (max):", normalizer[key].params_dict['input_stats']['max'])
                    print("Input Stats (mean):", normalizer[key].params_dict['input_stats']['mean'])
                    print("Input Stats (std):", normalizer[key].params_dict['input_stats']['std']) 
                     
                else:
                    normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                        self.replay_buffer[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]])

        # image
        for key in list(set(self.rgb_keys + self.extended_rgb_keys)):
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'][:, :self.shape_meta['action']['shape'][0]])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)   # (none,6,mone)
        obs_downsample_ratio = self.obs_downsample_ratio    # 3
        # logger.debug(f"T_slice : {T_slice}, obs_downsample_ratio: {obs_downsample_ratio}")
        # obs_path_init = f"/home/embodied-ai/mcy/reactive_diffusion_policy_umi/dataset_obs_log_init.txt"
        # with open(obs_path_init, "a") as w:
        #     w.write(f"tcp_pos: {data['left_robot_tcp_pos'].astype(np.float32)}\n")
        #     w.write(f"tcp_rot: {data['left_robot_rot_axis_angle'].astype(np.float32)}\n")
        obs_dict = dict()
        for key in self.rgb_keys:   #left_wrist_img
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice][::-obs_downsample_ratio][::-1],-1,1
                ).astype(np.float32) / 255.0
            # logger.info(f"obs_dict_{key} before shape: {obs_dict[key].shape}")
            if self.use_constant_rgb:   # 07/07
                obs_dict[key][...] = self.constant_rgb_value
                logger.info(f"obs_dict_with constant_rgb: {obs_dict[key[...]]}")
            # T,C,H,W
            # save ram
            if key not in self.rgb_keys:
                del data[key]
        for key in self.lowdim_keys:    #  ['left_robot_gripper_width', 'left_robot_tcp_pose', 'left_robot_tcp_wrench']
            if 'wrt' not in key:
                ###################### original ###################
                # obs_dict[key] = data[key][:, :self.shape_meta['obs'][key]['shape'][0]][T_slice][::-obs_downsample_ratio][::-1].astype(np.float32)
                # # save ram
                # if key not in self.extended_lowdim_keys:
                #     del data[key]
                ##############################################
                ############# 08/07 3dtacdex3d #############
                if key.endswith('tcp_wrench'):
                    # force_3d = data[key].astype(np.float32)
                    force_3d= data[key][:, :self.shape_meta['obs'][key]['shape'][0]][T_slice][::-obs_downsample_ratio][::-1].astype(np.float32)
                    # logger.debug(f"force_3d shape:{force_3d.shape}")
                    force_3d = force_3d.reshape(-1,16,3)
                    # logger.debug(f"force_3d shape2:{force_3d.shape}")
                    normal_force_3d = (force_3d-self.mean)/self.std
                    tactile_point = forward_kinematics(coords_type="full", coords_space="base")
                    tactile_point = np.array(tactile_point)
                    tactile_point_xyz = tactile_point[0,:,:3]
                    T = normal_force_3d.shape[0]
                    tactile_point_xyz_expanded = np.repeat(tactile_point_xyz[None, :, :], T, axis=0) 
                    matched_tactile_xyzfxfyfz = np.concatenate([tactile_point_xyz_expanded, normal_force_3d],axis=2)
                    # logger.info(f"matched tactile xyzfxfyfz shape: {matched_tactile_xyzfxfyfz.shape}")
                    data_batch, _, _, _ = data_to_gnn_batch(matched_tactile_xyzfxfyfz, edge_type='four+sensor')
                    # logger.debug(f"data_batch shape: {data_batch.x.shape}, edge_index shape: {data_batch.edge_index.shape}")
                    '''
                    data_batch.x shape: (32, 6)
                    edge_index shape: (2, 96)
                    '''
                    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    device = "cpu"
                    data_batch = data_batch.to(device)
                    latent_output = self.encoder(data_batch.x, data_batch.edge_index)
                    # logger.debug(f"latent_output shape: {latent_output.shape}")
                    latent_output_dim = 16*16
                    latent_output = latent_output.reshape(-1, 16, 16)  # (B, 16, 16)
                    latent_output = latent_output.reshape(-1, latent_output_dim)  # (B, 16*16)
                    obs_dict[key] = latent_output.cpu().numpy()  # Convert to numpy array (32 x 16, 16) = (B x 16,16)
                    # logger.debug(f"obs_dict[{key}] shape: {obs_dict[key].shape}")
                    # logger.debug(f"obs_dict[{key}] : {obs_dict[key]}")
                    # del data[key]
                else:
                    obs_dict[key] = data[key][:, :self.shape_meta['obs'][key]['shape'][0]][T_slice][::-obs_downsample_ratio][::-1].astype(np.float32)
                    # logger.debug(f"obs_dict[{key}] shape: {obs_dict[key].shape}")
                    if key not in self.extended_lowdim_keys:
                        del data[key]

        # inter-gripper relative action
        # obs_dict.update(get_inter_gripper_actions(obs_dict, self.lowdim_keys, self.transforms))
        # for key in ['left_robot_wrt_right_robot_tcp_pose', 'right_robot_wrt_left_robot_tcp_pose']:
        #     if key in obs_dict:
        #         obs_dict[key] = obs_dict[key][:, :self.shape_meta['obs'][key]['shape'][0]].astype(np.float32)
        
        extended_obs_dict = dict()
        for key in self.extended_rgb_keys:
            extended_obs_dict[key] = np.moveaxis(data[key],-1,1
                ).astype(np.float32) / 255.0
            # logger.info(f"extended_obs_dict[key] before constant_rgb: {extended_obs_dict[key][...]}")  # 07/07
            if self.use_constant_rgb:   # 07/07
                obs_dict[key][...] = self.constant_rgb_value
            del data[key]
        for key in self.extended_lowdim_keys:
            if 'wrt' not in key:
                ##### original #####
                # extended_obs_dict[key] = data[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]].astype(np.float32)
                # del data[key]
                #####################
                ############# 08/07 3dtacdex3d #############
                if key.endswith('tcp_wrench'):
                    # force_3d = data[key].astype(np.float32)
                    force_3d = data[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]].astype(np.float32)
                    # logger.debug(f"force_3d shape:{force_3d.shape}")
                    force_3d = force_3d.reshape(-1,16,3)
                    F = force_3d.sum(axis=1)  # sum over taxels
                    extended_obs_dict[key] = F
                    # logger.debug(f"force_3d : {F}")
                    # logger.debug(f"extended_obs_dict[key] shape: {extended_obs_dict[key].shape}")
                    # logger.debug(f"extended_obs_dict[key] : {extended_obs_dict[key]}")
                    del data[key]
                else:
                    extended_obs_dict[key] = data[key][:, :self.shape_meta['extended_obs'][key]['shape'][0]].astype(np.float32)
                    # logger.debug(f"obs_dict[{key}] shape: {extended_obs_dict[key].shape}")
                    del data[key]


        action = data['action'][:, :self.shape_meta['action']['shape'][0]].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
            # print("action: ", action)
            for key in extended_obs_dict:
                extended_obs_dict[key] = extended_obs_dict[key][self.n_latency_steps:]
        
        if self.relative_action:
            base_absolute_action = np.concatenate([
                obs_dict['left_robot_tcp_pose'][-1] if 'left_robot_tcp_pose' in obs_dict else np.array([]),
                obs_dict['right_robot_tcp_pose'][-1] if 'right_robot_tcp_pose' in obs_dict else np.array([])
            ], axis=-1)
            # print("action before relateive: ", action)
            # print("base_absolute_action: ", base_absolute_action)
            # time.sleep(5)
            action = absolute_actions_to_relative_actions(action, base_absolute_action=base_absolute_action)
            # print("relative action: ", action)

            # relative_tcp_obs 저장용 dict
            relative_tcp_obs_dict = {}
            absolute_tcp_obs_dict = {}

            if self.relative_tcp_obs_for_relative_action: #true
                for key in self.lowdim_keys:
                    if 'robot_tcp_pose' in key and 'wrt' not in key:
                        absolute_tcp_obs_dict[key] = obs_dict[key].tolist() #07/12
                        obs_dict[key]  = absolute_actions_to_relative_actions(obs_dict[key], base_absolute_action=base_absolute_action)
                        relative_tcp_obs_dict[key] = obs_dict[key].tolist() # 07/12

            # log_path = f"/home/embodied-ai/mcy/reactive_diffusion_policy_umi/relative_action_log.txt"  # ← 원하는 경로로 변경 가능 # 07/12
            # with open(log_path, "a") as f:  # append 모드
                # f.write(f"base_pos: {obs_dict['left_robot_tcp_pos']}\n")
                # f.write(f"base rot: {obs_dict['left_robot_rot_axis_angle']}\n")
                # f.write(f"base_absolute_action: {base_absolute_action.tolist()}\n")
                # for key, val in absolute_tcp_obs_dict.items():
                #     f.write(f"absolute: {key}: {val}\n")
                # for key, val in relative_tcp_obs_dict.items():
                #     f.write(f"relative: {key}: {val}\n")
                # f.write(f"relative_action: {action.tolist()}\n")
                # f.write("\n")

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action),
            'extended_obs': dict_apply(extended_obs_dict, torch.from_numpy)
        }
        return torch_data

#### 08/07 3dtacdex3d forward kinematics
point_per_sensor = len(XELA_USPA44_COORD)
def get_tactile_points(tactile_ori, tactile_points, link_pose, coords_type, coords_space):
    if coords_type == "full":
        local_points = tactile_ori + tactile_points # 각 taxel 상대 좌표 + 센서 원점
        rotation = link_pose[:3, :3]    
        translation = link_pose[:3, 3]
        real_points = (rotation @ local_points.T).T + translation   # 센서 기준 taxel 좌표를 로봇 base frame으로 변환 여기서는 link pose 고정이라 생각해 무시
        real_angle = R.from_matrix(rotation).as_euler("xyz").reshape(1, 3) / np.pi
        real_points = np.concatenate(
            [real_points, np.repeat(real_angle, point_per_sensor, axis=0)], axis=1
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
                    np.repeat(real_points, point_per_sensor, axis=0),
                    np.repeat(real_angle, point_per_sensor, axis=0),
                    tactile_points,
                ],
                axis=1,
            )
    return real_points
    
def forward_kinematics(coords_type="full",coords_space="base",):
    link_pose = np.eye(4)   # 센서가 고정된 pose로 가정
    coords_space = coords_space
    tactile_points = get_tactile_points(
        XELA_TACTILE_ORI_COORD,
        XELA_USPA44_COORD,
        link_pose,
        coords_type,
        coords_space,
    )
    return [tactile_points] # Return a list with a single element (16 taxel point 3d positions + angles) -> [16,6]

def test():
    import hydra
    from hydra import initialize, compose
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with initialize('../config'):
        cfg = hydra.compose('train_diffusion_unet_real_image_workspace',
                            overrides=['task=real_peel_image_gelsight_emb_absolute_12fps'])
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()

    for i in range(len(dataset)):
        data = dataset[i]

if __name__ == '__main__':
    test()
