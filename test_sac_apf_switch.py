import numpy as np
from ur5_drl_env.ur5_gym_env import UR5GymEnv
import rclpy

def test_sac_apf_switch():
    print("开始 SAC-APF 切换逻辑与奖励函数测试...")
    
    # 1. 实例化环境
    env = UR5GymEnv()
    
    # 2. 重置环境并获取初始状态
    obs, info = env.reset()
    print(f"环境已重置. 目标 EE 位置: {env.unwrapped.target_ee_position}")
    
    # 3. 模拟远距离情况
    dummy_sac_action = np.zeros(6, dtype=np.float32)
    
    print("\n--- 情况 1: 远距离测试 (预期使用 SAC 动作) ---")
    # 强制设置一个较远的距离 (例如 0.2m)
    env.unwrapped.current_joint_positions = np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(dummy_sac_action)
    
    dist_p = info['distance_p']
    apf_active = info['apf_active']
    
    if not apf_active:
        print(f"Step 1: Dist={dist_p:.4f}m -> Using SAC Action")
    else:
        print(f"Step 1: Dist={dist_p:.4f}m -> 异常: 不应在远距离激活 APF")

    # 4. 模拟近距离情况 (进入 APF 接管区)
    print("\n--- 情况 2: 近距离测试 (预期自动切换至 APF) ---")
    # 设置关节角度，使得末端位置非常接近目标 (例如 0.04m)
    env.unwrapped.current_joint_positions = env.unwrapped.target_joint_positions.copy()
    env.unwrapped.current_joint_positions += 0.02 # 制造微小偏差
    
    obs, reward, terminated, truncated, info = env.step(dummy_sac_action)
    
    dist_p = info['distance_p']
    apf_active = info['apf_active']
    
    if apf_active:
        print(f"Step 2: Dist={dist_p:.4f}m -> Switched to APF Action")
        print(f"当前奖励值 (包含精度奖励): {reward:.4f}")
    else:
        print(f"Step 2: Dist={dist_p:.4f}m -> 异常: 应在近距离激活 APF")

    # 5. 模拟成功到达目标
    print("\n--- 情况 3: 成功到达目标测试 ---")
    # 设置关节角度完全等于目标
    env.unwrapped.current_joint_positions = env.unwrapped.target_joint_positions.copy()
    
    obs, reward, terminated, truncated, info = env.step(dummy_sac_action)
    
    dist_p = info['distance_p']
    
    if terminated and dist_p < env.unwrapped.FINAL_GOAL_THRESHOLD:
        print(f"Episode Finished: Success=True, Final Error={dist_p:.4f}m")
        print(f"最终奖励值 (包含成功奖励): {reward:.4f}")
    else:
        print(f"Episode Finished: 异常: 未能正确识别成功状态")

    env.close()
    print("\n测试完成!")

if __name__ == "__main__":
    test_sac_apf_switch()
