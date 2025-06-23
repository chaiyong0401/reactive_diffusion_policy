import numpy as np
from typing import Tuple, Union
import transforms3d as t3d
from geometry_msgs.msg import Pose

def ros_pose_to_4x4matrix(pose: Pose) -> np.ndarray:
    # Convert ROS Pose message to 4x4 transformation matrix
    mat = np.eye(4)
    quat = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
    rot_mat = t3d.quaternions.quat2mat(quat)
    mat[:3, :3] = rot_mat
    mat[:3, 3] = np.array([pose.position.x, pose.position.y, pose.position.z])
    return mat

def ros_pose_to_6d_pose(pose: Pose) -> np.ndarray:
    # convert ROS Pose message to 6D pose (x, y, z, r, p, y)
    quat = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
    euler = t3d.euler.quat2euler(quat)
    trans = np.array([pose.position.x, pose.position.y, pose.position.z])
    return np.concatenate([trans, euler])

def pose_6d_to_pose_7d(pose: np.ndarray) -> np.ndarray:
    # convert 6D pose (x, y, z, r, p, y) to 7D pose (x, y, z, qw, qx, qy, qz)
    quat = t3d.euler.euler2quat(pose[3], pose[4], pose[5])
    return np.concatenate([pose[:3], quat])

def pose_7d_to_pose_6d(pose: np.ndarray) -> np.ndarray:
    # convert 7D pose (x, y, z, qw, qx, qy, qz) to 6D pose (x, y, z, r, p, y)
    quat = pose[3:]
    euler = t3d.euler.quat2euler(quat)
    return np.concatenate([pose[:3], euler])

def pose_7d_to_4x4matrix(pose: np.ndarray) -> np.ndarray:
    # convert 7D pose (x, y, z, qw, qx, qy, qz) to 4x4 transformation matrix
    mat = np.eye(4)
    mat[:3, :3] = t3d.quaternions.quat2mat(pose[3:])
    mat[:3, 3] = pose[:3]
    return mat

def pose_6d_to_4x4matrix(pose: np.ndarray) -> np.ndarray:
    # convert 6D pose (x, y, z, r, p, y) to 4x4 transformation matrix
    mat = np.eye(4)
    quat = t3d.euler.euler2quat(pose[3], pose[4], pose[5])
    mat[:3, :3] = t3d.quaternions.quat2mat(quat)
    mat[:3, 3] = pose[:3]
    return mat

def matrix4x4_to_pose_6d(mat: np.ndarray) -> np.ndarray:
    # convert 4x4 transformation matrix to 6D pose (x, y, z, r, p, y)
    quat = t3d.quaternions.mat2quat(mat[:3, :3])
    euler = t3d.euler.quat2euler(quat)
    trans = mat[:3, 3]
    return np.concatenate([trans, euler])

def ortho6d_to_rotation_matrix(ortho6d: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix from ortho6d representation
    """
    x_raw = ortho6d[:, 0:3]  # batch * 3
    y_raw = ortho6d[:, 3:6]  # batch * 3
    x = normalize_vector(x_raw)  # batch * 3
    z = np.cross(x, y_raw)  # batch * 3
    z = normalize_vector(z)  # batch * 3
    y = np.cross(z, x)  # batch * 3

    x = x[:, :, np.newaxis]
    y = y[:, :, np.newaxis]
    z = z[:, :, np.newaxis]

    matrix = np.concatenate((x, y, z), axis=2)  # batch * 3 * 3
    return matrix

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector (batch * 3)
    """
    v_mag = np.linalg.norm(v, axis=1, keepdims=True)  # batch * 1
    v_mag = np.maximum(v_mag, 1e-8)
    v = v / v_mag
    return v

def transform_point_cloud(pcd: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Transform a point cloud with 4x4 transform_matrix
    Parameters
    ----------
    pcd: (N, 6) or (N, 3) ndarray
    transform_matrix: (4, 4) ndarray
    """
    if pcd.shape[1] == 3: # (x, y, z)
        transformed_xyz = np.matmul(transform_matrix[:3, :3], pcd.T).T + transform_matrix[:3, 3]
        return transformed_xyz
    elif pcd.shape[1] == 6:  # (x, y, z, r, p, y)
        transformed_xyz = np.matmul(transform_matrix[:3, :3], pcd[:, :3].T).T + transform_matrix[:3, 3]
        return np.concatenate([transformed_xyz, pcd[:, 3:]], axis=1)
    else:
        raise NotImplementedError

def pose_6d_to_pose_9d(pose: np.ndarray) -> np.ndarray:
    """
    Convert 6D state to 9D state
    :param pose: np.ndarray (6,), (x, y, z, rx, ry, rz)
    :return: np.ndarray (9,), (x, y, z, rx1, rx2, rx3, ry1, ry2, ry3)
    """
    rot_6d = pose_6d_to_4x4matrix(pose)[:3, :2].T.flatten()
    return np.concatenate((pose[:3], rot_6d), axis=0)

def pose_3d_9d_to_homo_matrix_batch(pose: np.ndarray) -> np.ndarray:
    """
    Convert 3D / 9D states to 4x4 matrix
    :param pose: np.ndarray (N, 9) or (N, 3)
    :return: np.ndarray (N, 4, 4)
    """
    assert pose.shape[1] in [3, 9], "pose should be (N, 3) or (N, 9)"
    mat = np.eye(4)[None, :, :].repeat(pose.shape[0], axis=0)
    mat[:, :3, 3] = pose[:, :3]
    if pose.shape[1] == 9:
        mat[:, :3, :3] = ortho6d_to_rotation_matrix(pose[:, 3:9])
    return mat

def homo_matrix_to_pose_9d_batch(mat: np.ndarray) -> np.ndarray:
    """
    Convert 4x4 matrix to 9D state
    :param mat: np.ndarray (N, 4, 4)
    :return: np.ndarray (N, 9)
    """
    assert mat.shape[1:] == (4, 4), "mat should be (N, 4, 4)"
    pose = np.zeros((mat.shape[0], 9))
    pose[:, :3] = mat[:, :3, 3]
    pose[:, 3:9] = mat[:, :3, :2].swapaxes(1, 2).reshape(mat.shape[0], -1)
    return pose


import numpy as np
import scipy.spatial.transform as st

def pos_rot_to_mat(pos, rot):
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot

def pose_to_mat(pose):
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def transform_pose(tx, pose):
    """
    tx: tx_new_old
    pose: tx_old_obj
    result: tx_new_obj
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose

def transform_point(tx, point):
    return point @ tx[:3,:3].T + tx[:3,3]

def project_point(k, point):
    x = point @ k.T
    uv = x[...,:2] / x[...,[2]]
    return uv

def apply_delta_pose(pose, delta_pose):
    new_pose = np.zeros_like(pose)

    # simple add for position
    new_pose[:3] = pose[:3] + delta_pose[:3]

    # matrix multiplication for rotation
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose

def normalize(vec, tol=1e-7):
    return vec / np.maximum(np.linalg.norm(vec), tol)

def rot_from_directions(from_vec, to_vec):
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot

def normalize(vec, eps=1e-12):
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

def mat_to_rot6d(mat):
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose10d(mat):
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose10d_to_mat(d10):
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out
