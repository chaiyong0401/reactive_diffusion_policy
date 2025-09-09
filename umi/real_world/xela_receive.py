# 07/28
import time
import numpy as np
try:    # 07/28
    # from xela_server_ros2.msg._sensor_full import SensorFull  # type: ignore
    from xela_server_ros2.msg import SensStream  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    # SensorFull = None
    SensStream = None
try:
    import rclpy
    from rclpy.node import Node
except Exception:
    rclpy = None
import threading
import collections
from loguru import logger


class RosTactileListener:   
    def __init__(self, topic: str, horizon: int = 2, num_taxels: int = 16):
        self.topic = topic
        self.horizon = horizon
        self.num_taxels = num_taxels
        self.buffer = collections.deque(maxlen=horizon)
        if rclpy is not None:
            rclpy.init(args=None)
            self.node = rclpy.create_node('umi_tactile_listener')
            self.node.create_subscription(
                SensStream,
                topic,
                self._callback,
                1
            )
            threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()
        # elif rospy is not None:
        #     rospy.Subscriber(topic, Float32MultiArray, self._callback, queue_size=1)

    def _callback(self, msg: SensStream):
        # arr = np.array(msg.data, dtype=np.float32).reshape(self.num_taxels, 3)
        left_taxels = np.zeros(16, dtype=np.float32)
        timestamp = time.time()
        if rclpy is not None:
            timestamp = self.node.get_clock().now().nanoseconds / 1e9
            sensors = getattr(msg, 'sensors', [])
            for sensor in sensors:
                # taxels = getattr(sensor, 'taxels', [])
                taxels = getattr(sensor, 'forces', [])
                # values = [float(getattr(f,'z',0.0)) for f in taxels]  # only z-axis
                # values = values[:16]  # only z axis
                #####
                valuesz = [float(getattr(f,'z',0.0)) for f in taxels]
                valuesx = [float(getattr(f,'x',0.0)) for f in taxels]
                valuesy = [float(getattr(f,'y',0.0)) for f in taxels]
                valuesx = valuesx[:16]
                valuesy = valuesy[:16]
                valuesz = valuesz[:16]
                values = np.stack([valuesx, valuesy, valuesz], axis=-1)
                real_values = np.array(values, dtype=np.float32)
                left_taxels = real_values

        self.buffer.append((timestamp, left_taxels))

    def get_wrench(self) -> np.ndarray:
        if len(self.buffer) == 0:
            return np.zeros((self.horizon, self.num_taxels, 3), dtype=np.float32) 
            # return np.zeros((self.horizon, self.num_taxels * 3), dtype=np.float32) 
        # data = [w for _, w in list(self.buffer)]
        # while len(data) < self.horizon:
        #     data.insert(0, data[0])
        # return np.stack(data[-self.horizon:], axis=0)
        timestamps, values = zip(*list(self.buffer))
        values = list(values)
        while len(values) < self.horizon:
            values.insert(0, values[0])
            timestamps = (timestamps[0],) + timestamps  # 첫 timestamp 반복
        # logger.debug(f"Returning tactile data with timestamps: {timestamps[-self.horizon:]}")
        return list(timestamps[-self.horizon:]), np.stack(values[-self.horizon:], axis=0)