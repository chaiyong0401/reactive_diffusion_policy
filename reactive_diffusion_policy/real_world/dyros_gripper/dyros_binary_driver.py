# import rclpy
# from rclpy.node import Node
from dynamixel_sdk import PortHandler, PacketHandler
# from std_msgs.msg import Float32

# Motor and communication settings
DEVICENAME = "/dev/ttyUSB0"
# DEVICENAME = "/dev/ttyUSB9"
PROTOCOL_VERSION = 2
BAUDRATE = 57600

# Dynamixel XL330 memory addresses
CURRENT_BASED_POSITION_MODE = 5
ADDR_XL330_OPERATING_MODE = 11
ADDR_XL330_TORQUE_ENABLE = 64
ADDR_XL330_GOAL_POSITION = 116
ADDR_XL330_CURRENT_LIMIT = 38
ADDR_XL330_GOAL_CURRENT = 102
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0

ADDR_XL330_PRESENT_VELOCITY = 128
ADDR_XL330_PRESENT_POSITION = 132
# Current and position limits
CURRENT_LIMIT = 200
GOAL_CURRENT = 160

GOAL_POSITION_OPEN = 2048
GOAL_POSITION_CLOSE = 1024

GRIPPER_MAX_WIDTH =110 # reference from wsg_50 gripper width 

# Motor IDs
DXL_ID_1 = 1
DXL_ID_2 = 2
COMM_SUCCESS = 0

class DYROSBinaryDriver:
    def __init__(self):
        self.portHandler = PortHandler(DEVICENAME)
        self.packetHandler = PacketHandler(PROTOCOL_VERSION)

        # Open port and set baudrate
        if not self.portHandler.openPort():
            raise RuntimeError("Failed to open the port")
        if not self.portHandler.setBaudRate(BAUDRATE):
            raise RuntimeError("Failed to set the baudrate")

        # Initialize motors
        self.init_motors()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def init_motors(self):
        # Motor 1 initialization
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID_1, ADDR_XL330_TORQUE_ENABLE, TORQUE_DISABLE)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID_1, ADDR_XL330_OPERATING_MODE, CURRENT_BASED_POSITION_MODE)
        self.packetHandler.write2ByteTxRx(self.portHandler, DXL_ID_1, ADDR_XL330_CURRENT_LIMIT, CURRENT_LIMIT)
        self.packetHandler.write2ByteTxRx(self.portHandler, DXL_ID_1, ADDR_XL330_GOAL_CURRENT, GOAL_CURRENT)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID_1, ADDR_XL330_TORQUE_ENABLE, TORQUE_ENABLE)

        # Motor 2 initialization
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID_2, ADDR_XL330_TORQUE_ENABLE, TORQUE_DISABLE)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID_2, ADDR_XL330_OPERATING_MODE, CURRENT_BASED_POSITION_MODE)
        self.packetHandler.write2ByteTxRx(self.portHandler, DXL_ID_2, ADDR_XL330_CURRENT_LIMIT, CURRENT_LIMIT)
        self.packetHandler.write2ByteTxRx(self.portHandler, DXL_ID_2, ADDR_XL330_GOAL_CURRENT, GOAL_CURRENT)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID_2, ADDR_XL330_TORQUE_ENABLE, TORQUE_ENABLE)

    def set_goal_position(self, position: int):
        """Set goal position for the gripper"""
        result, error = self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID_1, ADDR_XL330_GOAL_POSITION, position)
        if result != COMM_SUCCESS or error != 0:
            raise RuntimeError(f"Motor ID {DXL_ID_1} error: {self.packetHandler.getTxRxResult(result)}")

        result, error = self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID_2, ADDR_XL330_GOAL_POSITION, position)
        if result != COMM_SUCCESS or error != 0:
            raise RuntimeError(f"Motor ID {DXL_ID_2} error: {self.packetHandler.getTxRxResult(result)}")

    def ack_fault(self):
        """Acknowledge and reset any fault conditions"""
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID_1, ADDR_XL330_TORQUE_ENABLE, TORQUE_DISABLE)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID_2, ADDR_XL330_TORQUE_ENABLE, TORQUE_DISABLE)
        self.init_motors()

    def homing(self):
        """Move the gripper to the open position"""
        self.set_goal_position(GOAL_POSITION_OPEN)

    def script_query(self):
        """Query the current position and state of the gripper"""
        raw_position, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, DXL_ID_1, ADDR_XL330_PRESENT_POSITION)
        velocity, _, _ = self.packetHandler.read4ByteTxRx(self.portHandler, DXL_ID_1, ADDR_XL330_PRESENT_VELOCITY)

        # Scale the raw position to match the WSG gripper width range -> gripper width scaled position(0-110)
        scaled_position = ((raw_position - GOAL_POSITION_CLOSE) / (GOAL_POSITION_OPEN - GOAL_POSITION_CLOSE)) * GRIPPER_MAX_WIDTH
        info = {
            'position': scaled_position,
            'velocity': velocity,
        }
        
        return info

    def script_position_pd(self, position: float, velocity: float):
        """Set position and velocity for the gripper"""
        # dynamixel Scale position to Dynamixel range (1024 ~ 2048)
        scaled_position = int(
            (position / GRIPPER_MAX_WIDTH) * (GOAL_POSITION_OPEN - GOAL_POSITION_CLOSE) + GOAL_POSITION_CLOSE
        )

        # 속도 제한 적용
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID_1, ADDR_XL330_PRESENT_VELOCITY, int(velocity))
        self.packetHandler.write4ByteTxRx(self.portHandler, DXL_ID_2, ADDR_XL330_PRESENT_VELOCITY, int(velocity))

        self.set_goal_position(scaled_position)
        
        info = {
            'position': scaled_position,
            'velocity': velocity,
        }
        return info


    def disable_torque(self):
        """Disable motor torque"""
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID_1, ADDR_XL330_TORQUE_ENABLE, TORQUE_DISABLE)
        self.packetHandler.write1ByteTxRx(self.portHandler, DXL_ID_2, ADDR_XL330_TORQUE_ENABLE, TORQUE_DISABLE)

    def close(self):
        """Release resources and close the port"""
        self.disable_torque()
        self.portHandler.closePort()

def test():
    import numpy as np
    import time

    # Dynamixel 드라이버 초기화
    driver = DYROSBinaryDriver()
    
    try:
        # 초기화 및 Fault Acknowledge
        print("Acknowledging faults...")
        driver.ack_fault()
        
        # Homing (Gripper 열림)
        print("Homing gripper...")
        driver.homing()
        time.sleep(1)

        # 테스트: 열림 -> 닫힘 -> 열림
        print("Testing open-close-open sequence...")
        T = 2  # 각 동작 지속 시간 (초)
        dt = 1 / 30  # 제어 주기 (초) - 30Hz

        # 열림 -> 닫힘
        print("Moving from open to close...")
        pos_open_to_close = np.linspace(GRIPPER_MAX_WIDTH, 0., int(T / dt))
        for target_position in pos_open_to_close:
            print(f"Target Position: {target_position:.2f} mm")
            driver.script_position_pd(position=target_position, velocity=50)

            info = driver.script_query()
            print(f"Gripper Position: {info['position']:.2f} mm, Gripper Velocity: {info['velocity']}")

            time.sleep(dt)

        # 닫힘 -> 열림
        print("Moving from close to open...")
        pos_close_to_open = np.linspace(0., GRIPPER_MAX_WIDTH, int(T / dt))
        for target_position in pos_close_to_open:
            print(f"Target Position: {target_position:.2f} mm")
            driver.script_position_pd(position=target_position, velocity=50)

            info = driver.script_query()
            print(f"Gripper Position: {info['position']:.2f} mm, Gripper Velocity: {info['velocity']}")

            time.sleep(dt)

        print("Test completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 드라이버 종료 및 리소스 해제
        print("Closing driver...")
        driver.close()

# PID 게인 설정 함수
def set_pid_gains(portHandler, packetHandler, dxl_id, p_gain, i_gain, d_gain):
    # Proportional Gain
    result, error = packetHandler.write2ByteTxRx(portHandler, dxl_id, 84, int(p_gain))
    if result != COMM_SUCCESS or error != 0:
        raise RuntimeError(f"Failed to set P-Gain: {packetHandler.getTxRxResult(result)}")

    # Integral Gain
    result, error = packetHandler.write2ByteTxRx(portHandler, dxl_id, 82, int(i_gain))
    if result != COMM_SUCCESS or error != 0:
        raise RuntimeError(f"Failed to set I-Gain: {packetHandler.getTxRxResult(result)}")

    # Derivative Gain
    result, error = packetHandler.write2ByteTxRx(portHandler, dxl_id, 80, int(d_gain))
    if result != COMM_SUCCESS or error != 0:
        raise RuntimeError(f"Failed to set D-Gain: {packetHandler.getTxRxResult(result)}")



def main(args=None):
    # Example usage
    driver = DYROSBinaryDriver()
    set_pid_gains(driver.portHandler, driver.packetHandler, DXL_ID_1, p_gain=15, i_gain=0, d_gain=3)
    set_pid_gains(driver.portHandler, driver.packetHandler, DXL_ID_2, p_gain=15, i_gain=0, d_gain=3)
    try:
        # test 함수 실행
        test()
    except KeyboardInterrupt:
        print("Test interrupted by user.")
    


if __name__ == '__main__':
    main()
