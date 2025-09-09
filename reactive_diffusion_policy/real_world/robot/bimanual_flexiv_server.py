import threading
from typing import List, Dict
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from loguru import logger

from reactive_diffusion_policy.real_world.robot.single_flexiv_controller import FlexivController
from reactive_diffusion_policy.common.data_models import (BimanualRobotStates, MoveGripperRequest,
                                                          TargetTCPRequest, ActionPrimitiveRequest)

import numpy as np
import zerorpc
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

from reactive_diffusion_policy.real_world.dyros_gripper.dyros_gripper_controller import DYROSController
from reactive_diffusion_policy.real_world.dyros_gripper.dyros_binary_driver import DYROSBinaryDriver
from reactive_diffusion_policy.common.space_utils import pose_6d_to_pose_7d, pose_6d_to_4x4matrix, matrix4x4_to_pose_6d, mat_to_pose, pose_to_mat, mat_to_pose10d
from typing import Any
from reactive_diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import sys
logger.add(sys.stderr, level="DEBUG")

tx_flangerot90_tip = np.identity(4)
# tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])
# tx_flangerot90_tip[:3, 3] = np.array([-0.0786, 0, 0.247])
tx_flangerot90_tip[:3, 3] = np.array([-0.0628, 0, 0.247]) # fixed 
# self.tx_flangerot90_tip[:3, 3] = np.array([0.0, 0.0628, 0.247]) 

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3,:3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3,:3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @ tx_flangerot90_tip # 좌표계 변환 후 translation
tx_tip_flange = np.linalg.inv(tx_flange_tip)

class FrankaInterface:
    # def __init__(self, ip='172.16.0.3', port=4242): # server(nuc)의 ip + port 
    def __init__(self, ip='192.168.0.25', port=4242): # local simulation
        logger.info(f"Start Frakainterface at {ip}:{port}")
        # self.frequency = 30
        # self.max_pos_speed = 0.5
        # self.mat_rot_speed = 1.0
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")

        # self.thread = None
        # self.keep_running = False
        # self.pose_interp = None
        # self.curr_pose = None
        # self.last_waypoint_time = None


    def get_ee_pose(self):  # pos + rotvec
        flange_pose = np.array(self.server.get_ee_pose())
        # logger.info(f"flange_pose: {flange_pose}")
        # logger.debug(f"pose_to_mat(flange_pose): {pose_to_mat(flange_pose)}")
        # logger.debug(f"tx_flange_tip = {self.tx_flange_tip}")
        tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
        flange_recover = mat_to_pose(pose_to_mat(tip_pose)@ tx_tip_flange)
        # logger.info(f"flange_pose_recover(tip->flange): {flange_recover}")
        # logger.info(f"flange_pose_9d: {mat_to_pose10d(pose_to_mat(flange_pose))}")
        # logger.debug(f"flange_pose_recover_9d: {mat_to_pose10d(pose_to_mat(tip_pose)@tx_tip_flange)}")
        # logger.debug(f"tip_pose_9d: {mat_to_pose10d(pose_to_mat(flange_pose) @ tx_flange_tip)}")
        # pos = flange_pose[:3] + np.array([0.247, 0.0, -0.0628]) #07/09
        # rot = flange_pose[3:]  # 그대로 유지
        # tip_pose = np.concatenate([pos, rot])

        # pos = tip_pose[:3] + np.array([0.009,0.0,0.006])
        # rot = tip_pose[3:]
        # tip_pose =np.concatenate([pos,rot])
        # logger.info(f"tip_pose: {tip_pose}")
        # logger.debug(f"flange pose: {flange_pose}")
        # logger.debug(f"tip_pose: {tip_pose}")
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
        # logger.debug(f"update_desired_ee_pose: {pose}")
        if isinstance(pose, list): 
            pose = np.array(pose)
        # pose = mat_to_pose(pose_to_mat(pose) @ tx_tip_flange)
        self.server.update_desired_ee_pose(pose.tolist())

    def terminate_current_policy(self):
        self.server.terminate_current_policy()

    def close(self):
        self.server.close()



class BimanualFlexivServer():
    """
    Bimanual Flexiv Server Class
    """
    # TODO: use UDP to respond
    def __init__(self,
                 host_ip="192.168.2.187",
                 port: int = 8092,
                 left_robot_ip="192.168.2.110",
                 right_robot_ip="192.168.2.111",
                 use_planner: bool = False
                 ) -> None:
        self.host_ip = host_ip
        self.port = port
        self.left_gripper = None
        self.left_robot = None
        # self.left_robot = FlexivController(local_ip=host_ip,
        #                                    robot_ip=left_robot_ip, )
        # self.right_robot = FlexivController(local_ip=host_ip,
        #                                     robot_ip=right_robot_ip, )

        # self.left_robot.robot.setMode(self.left_robot.mode.NRT_CARTESIAN_MOTION_FORCE)
        # self.right_robot.robot.setMode(self.right_robot.mode.NRT_CARTESIAN_MOTION_FORCE)


        # # open the gripper
        # self.left_robot.gripper.move(0.1, 10, 0)
        # self.right_robot.gripper.move(0.1, 10, 0)

        # self.tx_flangerot90_tip = np.identity(4)
        # # tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])
        # # tx_flangerot90_tip[:3, 3] = np.array([-0.0786, 0, 0.247])
        # self.tx_flangerot90_tip[:3, 3] = np.array([-0.0628, 0, 0.247]) # fixed 
        # # self.tx_flangerot90_tip[:3, 3] = np.array([0.0, 0.0628, 0.247])

        # self.tx_flangerot45_flangerot90 = np.identity(4)
        # self.tx_flangerot45_flangerot90[:3,:3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()

        # self.tx_flange_flangerot45 = np.identity(4)
        # self.tx_flange_flangerot45[:3,:3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()

        # self.tx_flange_tip = self.tx_flange_flangerot45 @ self.tx_flangerot45_flangerot90 @ self.tx_flangerot90_tip
        # self.tx_tip_flange = np.linalg.inv(self.tx_flange_tip)

        if use_planner:
            # TODO: support bimanual planner
            raise NotImplementedError
        else:
            self.planner = None

        self.app = FastAPI()
        # Start the receiving command thread
        self.setup_routes()
        self.setup_shutdown()
        
    def setup_shutdown(self):
        @self.app.on_event("shutdown")
        def shutdown_event():
            logger.info("Shutting down DYROSController loop...")
            self.left_gripper.stop_loop()
            # if self.left_robot:
            #     self.left_robot.stop_loop()

    def setup_routes(self): # 안전, 통신, 과부하 등 Fault 상황 해제 
        # @self.app.post('/clear_fault')
        # async def clear_fault() -> List[str]:
        #     if self.left_robot.robot.isFault():
        #         logger.warning("Fault occurred on left robot server, trying to clear ...")
        #         thread_left = threading.Thread(target=self.left_robot.clear_fault)
        #         thread_left.start()
        #     else:
        #         thread_left = None
        #     if self.right_robot.robot.isFault():
        #         logger.warning("Fault occurred on right robot server, trying to clear ...")
        #         thread_right = threading.Thread(target=self.right_robot.clear_fault)
        #         thread_right.start()
        #     else:
        #         thread_right = None
        #     # Wait for both threads to finish
        #     fault_msgs = []
        #     if thread_left is not None:
        #         thread_left.join()
        #         fault_msgs.append("Left robot fault cleared")
        #     if thread_right is not None:
        #         thread_right.join()
        #         fault_msgs.append("Right robot fault cleared")
        #     return fault_msgs

        @self.app.get('/get_current_robot_states')
        async def get_current_robot_states() -> BimanualRobotStates:

            # left_robot_state = self.left_robot.get_current_robot_states()
            # right_robot_state = self.right_robot.get_current_robot_states()
            left_robot_gripper_state = self.left_gripper.get_current_gripper_states()
            left_robot_gripper_state = left_robot_gripper_state * 0.001
            # logger.debug(f"left_robot_gripper_state in get_current_robot_states(): {left_robot_gripper_state}")

            left_robot_pose_state = self.left_robot.get_ee_pose()
            quat = R.from_rotvec(left_robot_pose_state[3:]).as_quat()  # (x, y, z, w)
            
            quat = np.roll(quat, 1)  # → (w, x, y, z)
            left_robot_pose = np.concatenate([left_robot_pose_state[:3], quat])
            # logger.info(f"left_robot_pose = {left_robot_pose}")

            return BimanualRobotStates(
                leftRobotTCP = left_robot_pose, #(pos + rotvec) -> (x,y,z,qw,qx,qy,qz)
                # leftRobotTCP = left_robot_pose_state, #(pos + rotvec)  # 07/11 6d pose
                rightRobotTCP=[0.0]*7,
                leftRobotTCPVel=[0.0]*6,
                rightRobotTCPVel=[0.0]*6,
                leftRobotTCPWrench=[0.0]*6,
                rightRobotTCPWrench=[0.0]*6,
                leftGripperState=[left_robot_gripper_state, 0.0],
                # leftGripperState=[0.0, 0.0],
                rightGripperState=[0.0, 0.0],
            )

        @self.app.post('/move_gripper/{robot_side}')
        async def move_gripper(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

            # robot_gripper = self.left_robot.gripper if robot_side == 'left' else self.right_robot.gripper
            # robot_gripper.move(request.width, request.velocity, request.force_limit)
            # return {
            #     "message": f"{robot_side.capitalize()} gripper moving to width {request.width} "
            #                f"with velocity {request.velocity} and force limit {request.force_limit}"}

            # self.left_gripper.move_to(0.01, duration=0.5)
            logger.debug(f"gripper_command: {request.width}")
            self.left_gripper.move_to(request.width, duration=10)
            return {
                "message": f"{robot_side.capitalize()} gripper moving to width {request.width} "
                           f"with velocity {request.velocity} and force limit {request.force_limit}"}

        @self.app.post('/move_gripper_force/{robot_side}')
        async def move_gripper_force(robot_side: str, request: MoveGripperRequest) -> Dict[str, str]:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

            # self.left_gripper.move_to(request.width, duration=0.5)


            # robot_gripper = self.left_robot.gripper if robot_side == 'left' else self.right_robot.gripper
            # # use force control mode to grasp
            # robot_gripper.grasp(request.force_limit)
            return {
                "message": f"{robot_side.capitalize()} gripper grasp with force limit {request.force_limit}"}

        # @self.app.post('/stop_gripper/{robot_side}')
        # async def stop_gripper(robot_side: str) -> Dict[str, str]:
        #     if robot_side not in ['left', 'right']:
        #         raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

        #     robot_gripper = self.left_robot.gripper if robot_side == 'left' else self.right_robot.gripper

        #     robot_gripper.stop()
        #     return {"message": f"{robot_side.capitalize()} gripper stopping"}

        @self.app.post('/move_tcp/{robot_side}')
        # async def move_tcp(robot_side: str, request: TargetTCPRequest) -> Dict[str, str]:
        async def move_tcp(robot_side: str, request: TargetTCPRequest) -> Dict[str, Any]:
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

            robot = self.left_robot 

            try:
                print(f"request.target_tcp= {request.target_tcp}") # (pos + rotvec)
                target_tcp_array = np.array(request.target_tcp)
                flange_pose = mat_to_pose(pose_to_mat(target_tcp_array) @ tx_tip_flange)
                # logger.debug(f"taget_tcp_array: {mat_to_pose10d(pose_to_mat(target_tcp_array))}")
                # logger.debug(f"taget_flagne_pose: {mat_to_pose10d(pose_to_mat(target_tcp_array)@ tx_tip_flange)}")
                

                # logger.debug(f"target_tcp_array_tip_pose: {target_tcp_array}")
                # logger.debug(f"update_desired_ee_flange_pose: {flange_pose}")
                robot.update_desired_ee_pose(np.array(flange_pose))   # 07/07 pose update
                
                # return {"message": f"{robot_side.capitalize()} robot moving to target tcp {request.target_tcp}"}
                return {"message": f"{robot_side.capitalize()} robot moving to target tcp", "target_tcp": request.target_tcp}
            except Exception as e:
                logger.exception("move_tcp failed")
                raise HTTPException(status_code=500, detail="update_desired_ee_pose failed")

            # robot.tcp_move(request.target_tcp)
            target_tcp_array = np.array(request.target_tcp)
            flange_pose = mat_to_pose(pose_to_mat(target_tcp_array)@ tx_tip_flange)
            # pos = target_tcp_array[:3] - np.array([0.247, 0.0, -0.0628])    # 07/09
            # rot = target_tcp_array[3:]
            # flange_pose = np.concatenate([pos, rot])
            robot.update_desired_ee_pose(np.array(flange_pose))   # 07/07 pose update
            # logger.debug(f"{robot_side.capitalize()} robot moving to target tcp {request.target_tcp}")
            return {"message": f"{robot_side.capitalize()} robot moving to target tcp {request.target_tcp}"}

        # @self.app.post('/execute_primitive/{robot_side}')
        # async def execute_primitive(robot_side: str, request: ActionPrimitiveRequest) -> Dict[str, str]:
        #     if robot_side not in ['left', 'right']:
        #         raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

        #     robot = self.left_robot if robot_side == 'left' else self.right_robot

        #     robot.execute_primitive(request.primitive_cmd)
        #     return {"message": f"{robot_side.capitalize()} robot executing primitive {request}"}

        @self.app.get('/get_current_tcp/{robot_side}')
        async def get_current_tcp(robot_side: str) -> List[float]:
            print("hello")
            if robot_side not in ['left', 'right']:
                raise HTTPException(status_code=400, detail="Invalid robot side. Use 'left' or 'right'.")

            robot = self.left_robot if robot_side == 'left' else self.right_robot

            try:
                # flange_pose = np.array(robot.get_ee_pose())
                # # logger.debug(f"robot version flange pose: {flange_pose}")
                # pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
                print("EE Pose:", robot.get_ee_pose())
                return pose.tolist()
            except Exception as e:
                print("Exception: ", e)
                raise HTTPException(status_code=500, detail=str(e))

            # return robot.get_current_tcp()
            return robot.get_ee_pose()

        # @self.app.post('/birobot_go_home')
        # async def birobot_go_home() -> Dict[str, str]:
        #     if self.planner is None:
        #         return {"message": "Planner is not available"}
        #     self.left_robot.robot.setMode(self.left_robot.mode.NRT_JOINT_POSITION)
        #     self.right_robot.robot.setMode(self.right_robot.mode.NRT_JOINT_POSITION)

        #     current_q = self.left_robot.get_current_q() + self.right_robot.get_current_q()
        #     waypoints = self.planner.getGoHomeTraj(current_q)

        #     for js in waypoints:
        #         print(js)
        #         self.left_robot.move(js[:7])
        #         self.right_robot.move(js[7:])
        #         time.sleep(0.01)

        #     self.left_robot.robot.setMode(self.left_robot.mode.NRT_CARTESIAN_MOTION_FORCE)
        #     self.right_robot.robot.setMode(self.right_robot.mode.NRT_CARTESIAN_MOTION_FORCE)
        #     return {"message": "Bimanual robots have gone home"}

    def run(self):
        logger.info(f"Start Bimanual Robot Fast-API Server at {self.host_ip}:{self.port}")
        # uvicorn.run(self.app, host=self.host_ip, port=self.port, log_level="critical")

         ##### dyros gripper##
        self.left_gripper = DYROSController(use_meters = True)
        self.left_gripper.connect()
        self.left_gripper.start_loop()
        ##################################
        self.left_robot = FrankaInterface(ip='10.0.0.2', port=4242)
        # self.left_robot.connect()
        # self.left_robot.start_loop()
        # print("EE Pose", self.left_robot.get_ee_pose())

        # example_tcp = [0,0,0,0,0,0]
        # self.left_robot.update_desired_ee_pose(example_tcp)
        config = uvicorn.Config(app=self.app, host=self.host_ip, port=self.port, log_level="debug")
        # config = uvicorn.Config(app=self.app, host="0.0.0.0", port=self.port, log_level="debug")
        logger.info(f"Start Bimanual Robot Fast-API Server at {self.host_ip}:{self.port}")
        server = uvicorn.Server(config=config)
        
        server.run()

def main():
    from hydra import initialize, compose
    from hydra.utils import instantiate

    with initialize(config_path='../../../config', version_base="1.3"):
        # config is relative to a module
        cfg = compose(config_name="bimanual_two_realsense_one_gelslim")

    robot_server = instantiate(cfg.robot_server)
    robot_server.run()


if __name__ == "__main__":
    main()