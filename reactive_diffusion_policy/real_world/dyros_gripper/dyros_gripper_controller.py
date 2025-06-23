import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from reactive_diffusion_policy.common.precise_sleep import precise_wait
from reactive_diffusion_policy.real_world.dyros_gripper.dyros_binary_driver import DYROSBinaryDriver
from reactive_diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import threading
from loguru import logger

class DYROSController:
    def __init__(self,
            frequency=30,
            home_to_open=True,
            move_max_speed=200.0,
            receive_latency=0.0,
            use_meters=False,
            verbose=False
            ):
        # super().__init__(name="DYROSController")
        self.frequency = frequency
        self.home_to_open = home_to_open
        self.move_max_speed = move_max_speed
        self.receive_latency = receive_latency
        self.scale = 1000.0 if use_meters else 1.0
        self.verbose = verbose
        self.thread = None
        # self.port_lock = threading.Lock()

        
       # 초기 상태
        self.pose_interp = None
        self.curr_pos = 0.0
        self.last_waypoint_time = None

    # ========= launch method ===========
    def start_loop(self):
        self.keep_running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop_loop(self):
        self.keep_running = False
        print("call stop_loop")
        if self.thread:
            self.thread.join()
        self.driver.close()
        logger.info("[DYROSController] Stopped")

    def connect(self):
        self.driver = DYROSBinaryDriver()
        self.driver.ack_fault()
        self.driver.homing()
        time.sleep(2)
        info = self.driver.script_query()
        self.curr_pos = info['position']
        now = time.monotonic()
        self.last_waypoint_time = now
        self.pose_interp = PoseTrajectoryInterpolator(
            times=[now],
            poses=[[self.curr_pos, 0, 0, 0, 0, 0]]
        )
        logger.info(f"[DYROSController] Connected, initial pos: {self.curr_pos}")
    
    def get_current_gripper_states(self):
        return self.curr_pos
    
    # ========= main loop in process ============
    def _run(self):
        try:
            while self.keep_running:
                # command gripper
                info = self.driver.script_query()
                self.curr_pos = info['position']
                # logger.info(f"[DYROSController]run1 pos: {info['position']}")
                t_now = time.monotonic()
                dt = 1 / self.frequency
                t_target = t_now
                target_pos = self.pose_interp(t_target)[0]
                # print(f"target_pos:{target_pos}")
                target_vel = (target_pos - self.pose_interp(t_target - dt)[0]) / dt
                # print('controller', target_pos, target_vel)

                # 특정 위치(target_pos)와 속도(target_vel)를 목표로 그리퍼를 제어하는 명령을 하드웨어로 전송 -> dyros gripper는 오직 위치제어만
                # 반환값 info는 명령 수행 결과에 대한 정보를 포함
                info = self.driver.script_position_pd(
                    position=target_pos, velocity = 50)
                # time.sleep(dt)
                # time.sleep(1e-3)

                # Interruption-aware sleep
                # info = self.driver.script_query()
                # logger.info(f"[DYROSController]run2 pos: {info['position']}")
                sleep_interval = 0.01
                slept = 0
                while self.keep_running and slept < dt:
                    time.sleep(sleep_interval)
                    slept += sleep_interval
                
                # execute commands
                # 큐에서 받은 명령을 해석하고 실행 (Shutdown 명령, 웨이포인트 명령, 경로 재설정)
                # 명령 처리 부분은 local에서 돌아가는 로직 -> 제어 명령을 생성(목표 위치(target_pos)와 속도(target_vel)를 계산) -> info = wsg.script_position_pd(하드웨어와의 통신)
        except Exception as e:
            logger.exception(f"[Dyroscontroller] exception in run loop: {e}")
        finally:
            if hasattr(self, 'driver'):
                self.driver.close()
            logger.info("[Dyroscontroller] run loop excited") 

    def move_to(self, pos: float, duration: float):
        target_pos = pos * self.scale
        now = time.monotonic()
        target_time = now + duration
        self.pose_interp = self.pose_interp.schedule_waypoint(
            pose=[target_pos, 0, 0, 0, 0, 0],
            time=target_time,
            max_pos_speed=self.move_max_speed,
            max_rot_speed=self.move_max_speed,
            curr_time=now,
            last_waypoint_time=self.last_waypoint_time
        )
        self.last_waypoint_time = target_time
        logger.info(f"New target scheduled: {target_pos} at {duration}s")