from typing import Optional
import numpy as np
import numba
from reactive_diffusion_policy.common.replay_buffer import ReplayBuffer
from loguru import logger

@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset

            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask

def get_finetune_train_mask(val_mask, finetune_ratio, seed=0):
    n_val = np.sum(val_mask)
    n_episodes = len(val_mask)
    val_ratio = (n_val / n_episodes)
    assert finetune_ratio < val_ratio, f"finetune_ratio {finetune_ratio} should be less than val_ratio {val_ratio}"

    finetune_mask = np.zeros(n_episodes, dtype=bool)
    # have at least 1 episode for finetune validation, and at least 1 episode for finetune train
    val_idxs = np.nonzero(val_mask)[0]
    n_finetune = min(max(1, round(n_episodes * finetune_ratio)), n_val-1)
    rng = np.random.default_rng(seed=seed)
    finetune_idxs = rng.choice(val_idxs, size=n_finetune, replace=False)
    finetune_mask[finetune_idxs] = True
    return finetune_mask

def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        ### UMI MCY 07/24 ###
        self.key_horizon_action = 32
        self.key_down_sample_steps_action = 1
        # # load gripper_width
        # gripper_width = replay_buffer['robot0_gripper_width'][:, 0]
        # gripper_width_threshold = 0.08
        # self.repeat_frame_prob = repeat_frame_prob

        # # create indices, including (current_idx, start_idx, end_idx)
        # indices2 = list()
        # for i in range(len(episode_ends)):
        #     before_first_grasp = True # initialize for each episode
        #     if episode_mask is not None and not episode_mask[i]:
        #         # skip episode
        #         continue
        #     start_idx = 0 if i == 0 else episode_ends[i-1]
        #     end_idx = episode_ends[i]
        #     if max_duration is not None:
        #         end_idx = min(end_idx, max_duration * 60)
        #     for current_idx in range(start_idx, end_idx):
        #         if not action_padding and end_idx < current_idx + (self.key_horizon['action'] - 1) * self.key_down_sample_steps['action'] + 1:
        #             continue
        #         if gripper_width[current_idx] < gripper_width_threshold:
        #             before_first_grasp = False
        #         indices2.append((current_idx, start_idx, end_idx, before_first_grasp))
        # self.indices2 = indices2
        # # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key == 'action':
                action_horizon = self.key_horizon_action
                action_down_sample_steps = self.key_down_sample_steps_action
                # downsample 시작 시점
                current_idx = buffer_start_idx  # 또는 다른 기준 시점이 있다면 수정
                slice_end = min(
                    buffer_end_idx,
                    current_idx + (action_horizon - 1) * action_down_sample_steps + 1
                )
                sample = input_arr[current_idx:slice_end:action_down_sample_steps]
                # padding if needed
                if sample.shape[0] < action_horizon:
                    padding = np.repeat(sample[-1:], action_horizon - sample.shape[0], axis=0)
                    sample = np.concatenate([sample, padding], axis=0)
                result[key] = sample
                # logger.debug(f"action sample: {sample}")
                continue
            if key not in self.key_first_k: # left_robot_tcp_pos, left_robot_rot_axis_angle, left_robot_tcp_wrench
                # logger.info(f"key: {key}")
                sample = input_arr[buffer_start_idx:buffer_end_idx]
                # logger.debug(f"key not in self.key_first_k: {sample}")
            else:   # left_robot_gripper_width, left_robot_tcp_pose -> 8 sample
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result

        ### MCY UMI 07/24 ###
        # current_idx, start_idx, end_idx, before_first_grasp = self.indices2[idx]

        # result = dict()

        # obs_keys = self.rgb_keys + self.lowdim_keys

        # # observation
        # for key in obs_keys:
        #     input_arr = self.replay_buffer[key][:]
        #     this_horizon = self.key_horizon[key]
        #     this_latency_steps = self.key_latency_steps[key]
        #     this_downsample_steps = self.key_down_sample_steps[key]
            
        #     if key in self.rgb_keys:
        #         assert this_latency_steps == 0
        #         num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)
        #         slice_start = current_idx - (num_valid - 1) * this_downsample_steps

        #         output = input_arr[slice_start: current_idx + 1: this_downsample_steps]
        #         assert output.shape[0] == num_valid
                
        #         # solve padding
        #         if output.shape[0] < this_horizon:
        #             padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
        #             output = np.concatenate([padding, output], axis=0)
        #     else:
        #         idx_with_latency = np.array(
        #             [current_idx - idx * this_downsample_steps + this_latency_steps for idx in range(this_horizon)],
        #             dtype=np.float32)
        #         idx_with_latency = idx_with_latency[::-1]
        #         idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
        #         interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
        #         interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

        #         if 'rot' in key:
        #             # rotation
        #             rot_preprocess, rot_postprocess = None, None
        #             if key.endswith('quat'):
        #                 rot_preprocess = st.Rotation.from_quat
        #                 rot_postprocess = st.Rotation.as_quat
        #             elif key.endswith('axis_angle'):
        #                 rot_preprocess = st.Rotation.from_rotvec
        #                 rot_postprocess = st.Rotation.as_rotvec
        #             else:
        #                 raise NotImplementedError
        #             slerp = st.Slerp(
        #                 times=np.arange(interpolation_start, interpolation_end),
        #                 rotations=rot_preprocess(input_arr[interpolation_start: interpolation_end]))
        #             output = rot_postprocess(slerp(idx_with_latency))
        #         else:
        #             interp = si.interp1d(
        #                 x=np.arange(interpolation_start, interpolation_end),
        #                 y=input_arr[interpolation_start: interpolation_end],
        #                 axis=0, assume_sorted=True)
        #             output = interp(idx_with_latency)
                
        #     result[key] = output

        # # repeat frame before first grasp
        # if self.repeat_frame_prob != 0.0:
        #     if before_first_grasp and random.random() < self.repeat_frame_prob:
        #         for key in obs_keys:
        #             result[key][:-1] = result[key][-1:]

        # # aciton
        # input_arr = self.replay_buffer['action']
        # action_horizon = self.key_horizon['action']
        # action_latency_steps = self.key_latency_steps['action']
        # assert action_latency_steps == 0
        # action_down_sample_steps = self.key_down_sample_steps['action']
        # slice_end = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        # output = input_arr[current_idx: slice_end: action_down_sample_steps]
        # # solve padding
        # if not self.action_padding:
        #     assert output.shape[0] == action_horizon
        # elif output.shape[0] < action_horizon:
        #     padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
        #     output = np.concatenate([output, padding], axis=0)
        # result['action'] = output

        # return result
