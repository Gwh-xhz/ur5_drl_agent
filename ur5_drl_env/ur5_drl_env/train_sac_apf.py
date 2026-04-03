import gymnasium as gym
from stable_baselines3 import SAC
from .ur5_gym_env import UR5GymEnv  #from ur5_drl_env.ur5_gym_env import UR5GymEnv
import os
import numpy as np

def train_sac_apf():
    # 1. 实例化自定义的 SAC-APF 环境
    env = UR5GymEnv()

    # 2. 初始化 SAC 算法
    # 论文推荐使用 SAC，因为它在连续控制任务中具有更好的探索能力
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        learning_starts=1000,
        use_sde=True, # 使用状态依赖探索以获得更平滑的动作
    )

    # 3. 课程学习 (Curriculum Learning) 配置
    # 随着训练进行，动态调整环境中的阈值
    total_timesteps = 100000
    
    print(f"开始 SAC-APF 混合控制训练，总步数: {total_timesteps}...")

    # 4. 训练循环 (模拟课程学习骨架)
    num_stages = 5
    steps_per_stage = total_timesteps // num_stages
    
    # 初始阈值 0.1m，最终阈值 0.05m
    start_threshold = 0.1
    end_threshold = 0.05
    
    for stage in range(num_stages):
        # 课程学习逻辑：逐渐减小 APF 切换阈值
        # 线性插值计算当前阶段的阈值
        progress = stage / max(1, num_stages - 1)
        new_threshold = start_threshold - progress * (start_threshold - end_threshold)
        
        env.unwrapped.APF_SWITCH_THRESHOLD = new_threshold
        
        print(f"\n--- 进入阶段 {stage + 1}/{num_stages} ---")
        print(f"当前 APF 切换阈值: {env.unwrapped.APF_SWITCH_THRESHOLD:.3f} 米")
        
        model.learn(
            total_timesteps=steps_per_stage, 
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        # 保存阶段模型
        model.save(f"ur5_sac_apf_stage_{stage+1}")

    # 5. 最终模型保存
    model.save("ur5_sac_apf_final")
    print("\n训练完成，最终模型已保存。")

    # 6. 测试
    print("开始模型测试...")
    obs, info = env.reset()
    for i in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if info.get("apf_active"):
            print(f"Step {i}: APF 激活! 距离: {info['distance_p']:.4f}m")
        
        if terminated or truncated:
            print(f"Episode 结束. 最终距离: {info['distance_p']:.4f}m")
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    train_sac_apf()
