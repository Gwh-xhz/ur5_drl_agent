import gymnasium as gym
from stable_baselines3 import PPO
from ur5_drl_env.ur5_gym_env import UR5GymEnv
import os

def main():
    # 1. 实例化自定义的 UR5 环境
    env = UR5GymEnv()

    # 2. 初始化 PPO 算法
    # policy: MlpPolicy (多层感知机策略)
    # env: 上面创建的环境
    # verbose: 输出训练日志
    model = PPO("MlpPolicy", env, verbose=1)

    # 3. 设置训练总步数
    total_timesteps = 10000
    
    print(f"开始训练，总步数: {total_timesteps}...")
    
    # 4. 执行训练循环
    model.learn(total_timesteps=total_timesteps)

    # 5. 保存训练好的模型
    model_path = "ur5_ppo_model"
    model.save(model_path)
    print(f"训练完成，模型已保存至: {model_path}")

    # 6. 测试训练好的模型
    print("开始测试模型...")
    obs, info = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()
