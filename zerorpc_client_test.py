# 외부 클라이언트 예시
import zerorpc
import numpy as np
from scipy.spatial.transform import Rotation

server = zerorpc.Client(heartbeat=20)
server.connect("tcp://10.0.0.2:4242")
print(server._zerorpc_list())


# 서버의 get_flange_pose 함수를 원격으로 호출
pose_list = server.get_ee_pose()

if pose_list:
    # 서버로부터 받은 리스트를 NumPy 4x4 행렬로 변환
    # Franka는 column-major 순서이므로 order='F' 옵션이 중요합니다.
    # pose_list = server.get_ee_pose()
    # flange_pose_matrix = np.array(pose_list).reshape(4, 4, order='F')
    # position = flange_pose_matrix[:3, 3]
    # rotation_matrix = flange_pose_matrix[:3, :3]
    # r = Rotation.from_matrix(rotation_matrix)
    # rotvec = r.as_rotvec()
    # pose_6d = np.concatenate([position, rotvec])
    # print("--- Received Flange Pose ---")
    # print(np.round(flange_pose_matrix, 3)) # 소수점 3자리까지 출력
    # print(f"6D Pose Vector: {np.round(pose_6d, 4)}")
    # print("--------------------------")

    pose_list = server.get_ee_pose()
    print("--- Received Flange Pose ---")
    print(f"List: {pose_list}")
    print("--------------------------")

waypoints = []
with open("trajectory_with_velocities.txt") as f:
    for line in f:
        vals = list(map(float, line.strip().split()))
        print(vals)
        waypoints.append(vals)
sucess, msg = server.send_ee_trajectory(waypoints)
print(sucess,msg)
server.close()