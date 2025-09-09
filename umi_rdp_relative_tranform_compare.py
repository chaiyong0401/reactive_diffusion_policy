import numpy as np

from reactive_diffusion_policy.common.action_utils import \
    absolute_actions_to_relative_actions
from umi.common.pose_util import pose10d_to_mat, mat_to_pose10d
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
# (or `compute_pose_mat_rep` if that is the name in your library)

# ---------------------------------------------------------------------
# Example absolute actions: (T, 10), where each row is [x,y,z,rot6d]
# ---------------------------------------------------------------------
absolute_actions = np.array([
    [0.5, 0.0, 0.2, 1, 0, 0, 0, 1, 0],
    [0.6, 0.0, 0.2, 1, 0, 0, 0, 1, 0],
    [0.6, 0.1, 0.2, 1, 0, 0, 0, 1, 0],
], dtype=np.float32)

base_action = absolute_actions[0]        # reference pose

# --- Method 1: use `absolute_actions_to_relative_actions`
rel1 = absolute_actions_to_relative_actions(
    absolute_actions, base_absolute_action=base_action
)

# --- Method 2: use pose matrices with `convert_pose_mat_rep`
abs_mat = pose10d_to_mat(absolute_actions)
base_mat = abs_mat[0]
rel_mat = convert_pose_mat_rep(
    abs_mat,
    base_pose_mat=base_mat,
    pose_rep="relative",
    backward=False
)
rel2 = mat_to_pose10d(rel_mat)

# --- Compare the two results
print("Method1 result:\n", rel1)
print("Method2 result:\n", rel2)
print("Are they close?", np.allclose(rel1, rel2, atol=1e-6))



from umi.common.pose_util import pose_to_mat, mat_to_pose
# 동일한 기능을 하는 함수가 common/space_utils.py에도 존재

def rel_pose_axis_angle(pos, rot_vec, base_pos, base_rot_vec):
    """pos + rotation_axis_vector 를 받아 상대 pose 계산"""
    pose = np.concatenate([pos, rot_vec], axis=-1)               # (6,)
    base_pose = np.concatenate([base_pos, base_rot_vec], axis=-1)

    base_mat = pose_to_mat(base_pose)            # (4,4) matrix
    cur_mat = pose_to_mat(pose)
    rel_mat = np.linalg.inv(base_mat) @ cur_mat  # T_base^cur

    rel_pose = mat_to_pose(rel_mat)              # (6,) : xyz + axis-angle
    return rel_pose

pose = np.array([1, 2, 3, 0.1, 0.2, 0.3])  # translation + rotvec
T = pose_to_mat(pose)
print(T)