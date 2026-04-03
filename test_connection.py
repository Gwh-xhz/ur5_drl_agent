import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time
import numpy as np

class ConnectionTester(Node):
    def __init__(self):
        super().__init__('connection_tester')
        
        # 定义标准的关节名称列表（与报告一致）
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]
        
        # 1. 订阅 /joint_states
        self.joint_state_received = False
        self.current_pos = np.zeros(6)
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)
        
        # 2. 发布控制命令 (轨迹控制器)
        self.publisher = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10)
        
        self.get_logger().info('测试节点已启动，正在等待 /joint_states 数据...')

    def listener_callback(self, msg):
        try:
            name_to_idx = {name: i for i, name in enumerate(msg.name)}
            all_found = True
            for i, target_name in enumerate(self.joint_names):
                if target_name in name_to_idx:
                    idx = name_to_idx[target_name]
                    self.current_pos[i] = msg.position[idx]
                else:
                    all_found = False
            
            if all_found and not self.joint_state_received:
                self.get_logger().info('已成功匹配并获取所有 6 个关节的数据！')
                self.joint_state_received = True
        except Exception as e:
            self.get_logger().error(f'处理回调时出错: {e}')

    def send_position_command(self, positions):
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        
        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in positions]
        point.time_from_start = Duration(sec=1, nanosec=0) # 1秒内到达
        
        traj_msg.points = [point]
        self.publisher.publish(traj_msg)

def main(args=None):
    rclpy.init(args=args)
    tester = ConnectionTester()

    # 等待并打印关节状态
    start_time = time.time()
    while rclpy.ok() and not tester.joint_state_received:
        rclpy.spin_once(tester, timeout_sec=0.1)
        if time.time() - start_time > 5.0:
            print("超时：未接收到完整的 6 关节数据。")
            break

    if tester.joint_state_received:
        print("-" * 30)
        print(f"初始关节位置: {tester.current_pos}")
        
        # 目标位置：将 shoulder_lift_joint 移动到 -0.5 rad
        target_pos = list(tester.current_pos)
        target_pos[1] -= 0.5 
        
        print(f"发送位置指令：将 shoulder_lift_joint 移动到 {target_pos[1]} rad...")
        tester.send_position_command(target_pos)
        
        # 等待运动完成
        time.sleep(2.0)
        
        rclpy.spin_once(tester, timeout_sec=0.5)
        print(f"最终关节位置: {tester.current_pos}")
        print("-" * 30)

    tester.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
