import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R
from umi.common.pose_util import pose_to_mat, mat_to_pose

tx_flangerot90_tip = np.identity(4)
# tx_flangerot90_tip[:3, 3] = np.array([-0.0826, 0, 0.257])
tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.237])
# tx_flangerot90_tip[:3, 3] = np.array([0, 0.05, 0.275])
# tx_flangerot90_tip[:3, 3] = np.array([-0.0628, 0, 0.247]) # fixed 
# tx_flangerot90_tip[:3, 3] = np.array([-0.0828, 0, 0.237])

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3,:3] = R.from_euler('x', [np.pi/2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3,:3] = R.from_euler('z', [np.pi/4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)


# # 텍스트 파일에서 데이터를 불러옵니다.
# try:
#     df = pd.read_csv('waypoints_aligned_quat.txt', comment='#', delim_whitespace=True, header=None)
#     df.columns = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
# except Exception as e:
#     # 파일을 불러오는 데 오류가 발생하면, 수동으로 파싱합니다.
#     with open('waypoints_aligned_quat.txt', 'r') as f:
#         lines = f.readlines()
    
#     data = []
#     for line in lines:
#         if line.startswith('#'):
#             continue
#         parts = line.split()
#         if len(parts) == 8:
#             data.append([float(p) for p in parts])
    
#     df = pd.DataFrame(data, columns=['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

# quaternions = df[['qx', 'qy', 'qz', 'qw']].to_numpy()
# positions = df[['x', 'y', 'z']].to_numpy()
# time = df['t'].to_numpy()
# r = R.from_quat(quaternions)

# rotation_matrices = r.as_matrix()
# rotation_rotvec = r.as_rotvec()

# pose = np.hstack((positions, rotation_rotvec))
# flange_pose = mat_to_pose(pose_to_mat(pose) @ tx_tip_flange)    
# quat_wp = R.from_rotvec(flange_pose[:,3:]).as_quat()

# rotation_matrices = R.from_rotvec(flange_pose[:,3:]).as_matrix()
# rotation_matrices_flat = rotation_matrices.reshape(-1, 9)

# time_column = time.reshape(-1, 1)
# print(flange_pose[1,:3])
# print(quat_wp[1,:])
# output_data = np.hstack((time_column, flange_pose[:, :3], quat_wp))

# np.savetxt('waypoints_flange_aligned_rotvec.txt', output_data)
# 텍스트 파일에서 데이터를 불러옵니다.
try:
    df = pd.read_csv('actual_aligned_quat.txt', comment='#', delim_whitespace=True, header=None)
    df.columns = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
except Exception as e:
    # 파일을 불러오는 데 오류가 발생하면, 수동으로 파싱합니다.
    with open('actual_aligned_quat.txt', 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) == 8:
            data.append([float(p) for p in parts])
    
    df = pd.DataFrame(data, columns=['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

quaternions = df[['qx', 'qy', 'qz', 'qw']].to_numpy()
positions = df[['x', 'y', 'z']].to_numpy()
time = df['t'].to_numpy()
r = R.from_quat(quaternions)

rotation_matrices = r.as_matrix()
rotation_rotvec = r.as_rotvec()

pose = np.hstack((positions, rotation_rotvec))
flange_pose = mat_to_pose(pose_to_mat(pose) @ tx_tip_flange)    
quat_wp = R.from_rotvec(flange_pose[:,3:]).as_quat()

rotation_matrices = R.from_rotvec(flange_pose[:,3:]).as_matrix()
rotation_matrices_flat = rotation_matrices.reshape(-1, 9)

time_column = time.reshape(-1, 1)
print(flange_pose[1,:3])
print(quat_wp[1,:])
output_data = np.hstack((time_column, flange_pose[:, :3], quat_wp))

np.savetxt('actual_flange_aligned_rotvec.txt', output_data)

# 3D 플롯을 생성합니다.
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 시간에 따라 색상이 변하는 3D 산점도를 그립니다.
# 'viridis' 컬러맵을 사용하여 시간의 흐름을 표현합니다.
sc = ax.scatter(df['x'], df['y'], df['z'], c=df['t'], cmap='viridis', marker='o')

# 색상 막대를 추가하고 라벨을 설정합니다.
cbar = plt.colorbar(sc)
cbar.set_label('Time (s)')

# 각 축의 라벨과 그래프의 제목을 설정합니다.
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Position Plot with Time-based Coloring')
ax.grid(True)

# 플롯을 이미지 파일로 저장합니다.
plt.savefig('3d_position_plot_colored.png')
plt.close()

print("Color-coded 3D plot has been generated and saved as 3d_position_plot_colored.png")


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 데이터 불러오기 ---
try:
    # 'Planned' 경로 데이터 불러오기
    df_planned = pd.read_csv('waypoints_aligned_quat.txt', comment='#', delim_whitespace=True, header=None)
    df_planned.columns = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
except FileNotFoundError:
    df_planned = None
except Exception as e:
    df_planned = None

try:
    # 'Actual' 경로 데이터 불러오기
    df_actual = pd.read_csv('actual_aligned_quat.txt', comment='#', delim_whitespace=True, header=None)
    df_actual.columns = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
except FileNotFoundError:
    df_actual = None
except Exception as e:
    df_actual = None


# --- 3D 그래프 생성 ---
if df_planned is not None and df_actual is not None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 'Planned' 경로 플롯 (파란색 실선)
    # ax.plot(df_planned['x'], df_planned['y'], df_planned['z'], label='Planned Path', color='blue', linestyle='-')
    ax.scatter(df_planned['x'], df_planned['y'], df_planned['z'], c=df_planned['t'], cmap='viridis', marker='o')

    # 'Actual' 경로 플롯 (빨간색 점선)
    # ax.plot(df_actual['x'], df_actual['y'], df_actual['z'], label='Actual Path', color='red', linestyle='--')
    ax.scatter(df_actual['x'], df_actual['y'], df_actual['z'], c=df_actual['t'], cmap='viridis', marker='^')

    # 시작점과 끝점 표시
    ax.scatter(df_planned['x'].iloc[0], df_planned['y'].iloc[0], df_planned['z'].iloc[0], color='green', s=10, label='Start', marker='o')
    ax.scatter(df_planned['x'].iloc[-1], df_planned['y'].iloc[-1], df_planned['z'].iloc[-1], color='green', s=10, label='End', marker='X')
    ax.scatter(df_actual['x'].iloc[0], df_actual['y'].iloc[0], df_actual['z'].iloc[0], color='purple', s=10, label='Actual_Start', marker='o')
    ax.scatter(df_actual['x'].iloc[-1], df_actual['y'].iloc[-1], df_actual['z'].iloc[-1], color='purple', s=10, label='Actual_End', marker='X')


    # --- 그래프 설정 ---
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('Planned vs. Actual 3D Trajectory')
    ax.legend() # 범례 표시
    ax.grid(True)

    # --- 파일로 저장 ---
    plt.savefig('combined_3d_plot.png')
    plt.close()
    print("Combined 3D plot has been generated and saved as 'combined_3d_plot.png'")
else:
    print("데이터 파일 중 하나 이상을 불러오지 못해 그래프를 생성할 수 없습니다.")