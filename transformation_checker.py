import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

# 입력
pos_cur= torch.tensor([0.3606, 0.0502, 0.5672])
pos_des= torch.tensor([ 0.4451, -0.0926,  0.4389])
quat_cur= torch.tensor([ 0.9100, -0.3820, -0.1311,  0.0937])
quat_des= torch.tensor([ 0.9539, -0.0910,  0.2800,  0.0584])


# scipy용 quaternion (x, y, z, w)
r_cur = R.from_quat(quat_cur.numpy())
r_des = R.from_quat(quat_des.numpy())

# 각각의 pose를 4x4 행렬로
T_cur = np.eye(4)
T_cur[:3, :3] = r_cur.as_matrix()
T_cur[:3, 3] = pos_cur.numpy()

T_des = np.eye(4)
T_des[:3, :3] = r_des.as_matrix()
T_des[:3, 3] = pos_des.numpy()

# 변환: 현재 → 목표
T_desired_current = T_des @ np.linalg.inv(T_cur)
print(T_desired_current)