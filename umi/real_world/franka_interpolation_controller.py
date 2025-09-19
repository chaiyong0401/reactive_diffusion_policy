import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
import torch
from umi.common.pose_util import pose_to_mat, mat_to_pose
import zerorpc
from loguru import logger

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

tx_flangerot90_tip = np.identity(4)
# tx_flangerot90_tip[:3, 3] = np.array([-0.0826, 0, 0.257])
tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.237])
# tx_flangerot90_tip[:3, 3] = np.array([0, 0.05, 0.275])
# tx_flangerot90_tip[:3, 3] = np.array([-0.0628, 0, 0.247]) # fixed 
# tx_flangerot90_tip[:3, 3] = np.array([-0.0828, 0, 0.237])

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3,:3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3,:3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)

class FrankaInterface:  # server가 돌아가는 nuc <-> client가 돌아가는 desktop
    # def __init__(self, ip='172.16.0.3', port=4242): # server(nuc)의 ip + port 
    # def __init__(self, ip='127.0.0.1', port=4242): # local simulation
    # def __init__(self, ip='147.47.190.17', port=4242): # local mcy com ip 
    def __init__(self, ip='192.168.0.25', port=4242): # husky-rcu ip
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        flange_pose = np.array(self.server.get_ee_pose())
        tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
        return tip_pose
    
    def get_joint_positions(self):
        return np.array(self.server.get_joint_positions())
    
    def get_joint_velocities(self):
        return np.array(self.server.get_joint_velocities())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        self.server.move_to_joint_positions(positions.tolist(), time_to_go)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self.server.start_cartesian_impedance(
            Kx.tolist(),
            Kxd.tolist()
        )
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        self.server.update_desired_ee_pose(pose.tolist())

    def terminate_current_policy(self):
        self.server.terminate_current_policy()

    def close(self):
        self.server.close()

class FrankaInterface_cycontroller:
    def __init__(self, ip='10.0.0.2', port=4242):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")
    
    def get_ee_pose(self):
        flange_pose_list = self.server.get_ee_pose()
        flange_pose_matrix = np.array(flange_pose_list).reshape(4, 4, order='F')
        position = flange_pose_matrix[:3, 3]
        rotation_matrix = flange_pose_matrix[:3, :3]
        r = st.Rotation.from_matrix(rotation_matrix)
        rotvec = r.as_rotvec()
        pose_6d = np.concatenate([position, rotvec]).tolist()
        tip_pose = mat_to_pose(pose_to_mat(pose_6d) @ tx_flange_tip)
        return tip_pose
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        flange_pose = mat_to_pose(pose_to_mat(pose) @ tx_tip_flange)
        self.server.update_desired_ee_pose(flange_pose)

    def send_ee_trajectory(self, waypoints):
        waypoints_flange = []
        for wp in waypoints:
            wp = np.array(wp)
            flange_pose = mat_to_pose(pose_to_mat(wp[:6]) @ tx_tip_flange)
            wp_flange = np.concatenate([flange_pose, wp[6:]])
            waypoints_flange.append(wp_flange.tolist())
        return self.server.send_ee_trajectory(waypoints_flange)



class FrankaInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        robot_ip,
        robot_port=4242,
        frequency=1000,
        Kx_scale=1.0,
        Kxd_scale=1.0,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0
        ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FrankaPositionalController")
        # self.robot_ip = robot_ip
        # self.robot_port = robot_port
        # self.robot_ip = "192.168.0.25"
        self.robot_ip = "10.0.0.2"
        self.robot_port = 4242
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            # 'cmd': Command.SERVOL.value,
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),   # ActualTCPPose : key //  get_ee_pose: robot 객체 method 이름  
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd','get_joint_velocities'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,    # 최대 조회 갯수 
            get_time_budget=0.2,    # get 호출 시 데이터 복사 허용 최대 시간
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

        # plan_example = {
        #     'planned_time': np.array(0.0, dtype=np.float64),
        #     'planned_pose': np.zeros(6, dtype=np.float64),
        #     'segment_id': np.array(0, dtype=np.int64),
        # }
        # plan_rb = SharedMemoryRingBuffer.create_from_examples(
        #     shm_manager=shm_manager,
        #     examples=plan_example,
        #     get_max_k=4096,
        #     get_time_budget=0.2,
        #     put_desired_frequency=frequency
        # )
        # self.plan_buffer = plan_rb
        # self.plan_sample_dt = 0.01
            
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()   # 버퍼가 k개 이상 차 있으면 최대 k개, 아니면 차있는 만큼 state return 
    
    def get_planned(self, k=None, out=None):
        if k is None:
            return self.plan_buffer.get_all()
        else:
            return self.plan_buffer.get_last_k(k=k, out=out)
    

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        # start polymetis interface
        robot = FrankaInterface(ip='10.0.0.2', port=4242)
        # robot = FrankaInterface() # local computer simul
        try:
            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")
            
            print(robot.get_ee_pose())
            # init pose
            if self.joints_init is not None:
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )

            # main loop
            dt = 1. / self.frequency
            curr_pose = robot.get_ee_pose()

            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            # start franka cartesian impedance policy
            robot.start_cartesian_impedance(
                Kx=self.Kx,
                Kxd=self.Kxd
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            t_first =0
            segment_id = 0
            while keep_running:
                # send command to robot
                t_now = time.monotonic()
                tip_pose = pose_interp(t_now) # t_now의 보간된 tip 위치 call
                # logger.info(f"original tip_pose: {tip_pose}")
                # flange_pose = mat_to_pose(pose_to_mat(tip_pose) @ tx_tip_flange)
                # logger.debug(f"original flange_pose: {flange_pose}")
                # tip_pose[0] += 0.01  # 약간 더 forward (z축 방향)
                # logger.info(f"modified tip_pose: {tip_pose}") 
                # logger.debug(f"flange to tip: {tx_flange_tip}")
                flange_pose = mat_to_pose(pose_to_mat(tip_pose) @ tx_tip_flange)    
                # logger.debug(f"modified flange_pose: {flange_pose}")

                # send command to robot
                robot.update_desired_ee_pose(flange_pose)
                # print(f"robot_desired_ee_pose: {flange_pose}")
                # print(f"robot_ee_pose:{robot.get_ee_pose()}")
                t_second = time.time()
                ts_f = t_second - t_first
                # print(f"franka interpolation running time:{ts_f}")
                t_first = time.time()
                # update robot stae
                '''
                # state = {
                    'ActualTCPPose': np.array(robot.get_ee_pose()),
                    'ActualQ': np.array(robot.get_joint_positions()),
                    'ActualQd': np.array(robot.get_joint_velocities()),
                    }
                '''
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = np.array(getattr(robot, func_name)())


                    
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    # commands = self.input_queue.get_all()
                    # n_cmd = len(commands['cmd'])
                    # process at most 1 command per cycle to maintain frequency
                    # len_all_command = self.input_queue.get_all()
                    # print(f"len_all_command:{len_all_command}")
                    commands = self.input_queue.get_k(1)
                    # print(f"commands:{commands}")
                    n_cmd = len(commands['cmd'])    # 이 경우 n_cmd =1 인듯
                    # print(f"n_command:{n_cmd}")
                except Empty:
                    # print("n_cmd = 0")
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    # print(f"cmd : {cmd}")

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:   # 사용 x 
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(    # target_pose로 이동하는 경로 생성, 보간 이용 
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:    # 실제로 사용되는 코드 
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(    # waypoint 보간하여 pose_interp으로 저장
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time

                        # === 여기부터 추가: 막 추가된 "마지막 세그먼트" 샘플해서 송신 ===
                        # tt = pose_interp.times
                        # if tt.size >= 2:
                        #     t0, t1 = float(tt[-2]), float(tt[-1])
                        #     if t1 > t0:
                        #         tq = np.arange(t0, t1 + 1e-9, self.plan_sample_dt)
                        #         Xq = pose_interp(tq)
                        #         logger.debug(f"tq: {tq}")
                        #         # 단조시각 → 월클락
                        #         mono_to_wall = time.time() - time.monotonic()
                        #         for t_s, p_s in zip(tq, Xq):
                        #             self.plan_buffer.put({
                        #                 'planned_time': t_s + mono_to_wall,
                        #                 'planned_pose': p_s,
                        #                 'segment_id': segment_id
                        #             })
                        #         segment_id += 1
                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[FrankaPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            # manditory cleanup
            # terminate
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            robot.terminate_current_policy()
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
