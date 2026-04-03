import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy
from .apf_controller import APFController

class UR5GymEnv(gym.Env):
    """
    基于 SAC-APF 混合控制的 UR5 机械臂 Gymnasium 环境。
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(UR5GymEnv, self).__init__()

        # ROS 2 初始化
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('ur5_gym_env_node')

        # DH 参数
        dh_params = {
            'd': [0.089159, 0, 0, 0.10915, 0.09465, 0.0823],
            'a': [0, -0.425, -0.39225, 0, 0, 0],
            'alpha': [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
        }

        # 实例化 APF 控制器
        self.apf_controller = APFController(dh_params)

        # 关节名称
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # 动作空间：关节角度增量
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(6,), dtype=np.float32)

        # 观测空间
        obs_space_dims = 6 + 6 + 3 + 3 + 1 # q, dq, pos_ee, pos_goal, dist_p
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_space_dims,), dtype=np.float32)

        # 内部状态
        self.q_curr = np.zeros(6, dtype=np.float32)
        self.dq_curr = np.zeros(6, dtype=np.float32)
        self.pos_goal = np.zeros(3, dtype=np.float32)
        self.q_start = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float32)

        # SAC-APF 参数
        self.APF_SWITCH_THRESHOLD = 0.05
        self.FINAL_GOAL_THRESHOLD = 0.01
        self.APF_MAX_STEPS = 50

        # ROS 2 通信
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.joint_sub = self.node.create_subscription(JointState, '/joint_states', self._joint_state_callback, qos)
        self.trajectory_pub = self.node.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)

    def _joint_state_callback(self, msg):
        try:
            name_to_idx = {name: i for i, name in enumerate(msg.name)}
            for i, target_name in enumerate(self.joint_names):
                if target_name in name_to_idx:
                    idx = name_to_idx[target_name]
                    self.q_curr[i] = msg.position[idx]
                    self.dq_curr[i] = msg.velocity[idx]
        except Exception as e:
            self.node.get_logger().error(f'Joint state callback error: {e}')

    def _get_observation(self):
        pos_ee = self.apf_controller.get_ee_pose(self.q_curr)
        delta_pos = self.pos_goal - pos_ee
        dist_p = np.linalg.norm(delta_pos)
        return np.concatenate([self.q_curr, self.dq_curr, pos_ee, self.pos_goal, [dist_p]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 在有效工作空间内随机生成目标点
        while True:
            r = self.np_random.uniform(0.3, 0.8)
            theta = self.np_random.uniform(0, 2 * np.pi)
            h = self.np_random.uniform(0.1, 0.6)
            self.pos_goal = np.array([r * np.cos(theta), r * np.sin(theta), h])
            # 简单的可达性检查 (避免奇异或极端位置)
            if np.linalg.norm(self.pos_goal) > 0.4:
                break
        
        self._publish_command(self.q_start)
        rclpy.spin_once(self.node, timeout_sec=0.5) # 等待机械臂移动
        return self._get_observation(), {}

    def step(self, action):
        pos_ee = self.apf_controller.get_ee_pose(self.q_curr)
        dist_p = np.linalg.norm(self.pos_goal - pos_ee)

        if dist_p > self.APF_SWITCH_THRESHOLD:
            # RL 模式
            print(f"[RL] Dist: {dist_p:.4f}m -> Executing RL Action")
            q_target = self.q_curr + action
            self._publish_command(q_target)
            rclpy.spin_once(self.node, timeout_sec=0.1)
            
            obs = self._get_observation()
            new_dist_p = obs[-1]
            reward = (dist_p - new_dist_p) * 100.0 - 0.01 # 距离引导 + 步数惩罚
            done = False
        else:
            # APF 模式
            print(f"[SWITCH] Dist: {dist_p:.4f}m -> APF Start")
            q_apf = self.q_curr.copy()
            for i in range(self.APF_MAX_STEPS):
                delta_q = self.apf_controller.compute_joint_increment(q_apf, self.pos_goal)
                q_apf += delta_q
                self._publish_command(q_apf)
                rclpy.spin_once(self.node, timeout_sec=0.05)
                
                current_pos_apf = self.apf_controller.get_ee_pose(q_apf)
                dist_apf = np.linalg.norm(self.pos_goal - current_pos_apf)
                
                if dist_apf < self.FINAL_GOAL_THRESHOLD:
                    print(f"[APF] Success! Final Dist: {dist_apf:.4f}m")
                    reward = 10.0
                    done = True
                    break
            else: # 循环正常结束 (未 break)
                print(f"[APF] Failed! Final Dist: {dist_apf:.4f}m")
                reward = -5.0
                done = True
            obs = self._get_observation()

        return obs, reward, done, False, {}

    def _publish_command(self, q_target):
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        point = JointTrajectoryPoint()
        point.positions = [float(q) for q in q_target]
        point.time_from_start = Duration(sec=0, nanosec=100_000_000)
        traj_msg.points = [point]
        self.trajectory_pub.publish(traj_msg)

    def close(self):
        if self.node:
            self.node.destroy_node()
