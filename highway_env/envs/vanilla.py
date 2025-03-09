import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv,  DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os
import clip
from PIL import Image
import torch
import numpy as np



env = gym.make(
    "intersection-v1",
    config={
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0 ,0, 0.05],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": False,
        "target_speeds": [0, 2.5, 5],
        },
        "duration": 30,
        "collision_reward": -5,
        "high_speed_reward": 0.5,
        "arrived_reward": 10,
        "reward_speed_range": [4.0, 5.0],
        "initial_vehicle_count": 5,
        'simulation_frequency': 15,
        "spawn_probability": 0.2,
    },
    #render_mode = "human"
)


# model = PPO(
#     "CnnPolicy",
#     env,
#     learning_rate=5e-4,     
#     n_steps = 256,           
#     batch_size=64,           
#     n_epochs=10,             
#     gamma=0.95,              
#     gae_lambda=0.95,         
#     clip_range=0.2,          
#     ent_coef=0.01,           
#     vf_coef=0.5,             
#     max_grad_norm=0.5,       
#     verbose=1,
#     tensorboard_log="C:/Users/erfan/Desktop/ppo/",              
# )


model = DQN('CnnPolicy',
            env,
            policy_kwargs = dict(net_arch=[256, 256]),
            learning_rate = 5e-4,
            buffer_size = 10000,
            exploration_fraction = 0.20,
            learning_starts = 200,
            batch_size = 64,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            target_update_interval= 256,
            verbose=1,
            tensorboard_log="C:/Users/erfan/Desktop/logs/"
            )


# model.learn(total_timesteps = 6000)
# model.save("DQN_intersection_vanilla")

#model = PPO.load("PPO_intersection5")
model = DQN.load("DQN_intersection_vanilla")
env = gym.make(
    "intersection-v1",
    config={
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0 ,0, 0.05],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": False,
        "target_speeds": [0, 2.5, 5],
        },
        
        "duration": 30,
        "collision_reward": -5,
        "high_speed_reward": 0.5,
        "arrived_reward": 2,
        "reward_speed_range": [4.0, 5.0],
        "initial_vehicle_count": 5,
        'simulation_frequency': 15,
        "spawn_probability": 0.2,
    },
    render_mode = "human"
)


for i in range(10):
        obs, info = env.reset()
        done = False
        truncated = False
        rewards = 0
        time = 0
        proceed = 0
        print(f'episode: {i}  -------------------')
        while not (done or truncated):

            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            rewards += reward
        
        print("total reward:", rewards)    
  
# for episode in range(5):
#     state = env.reset()
#     terminated = False
#     truncated = False
#     print(episode)
#     while not (terminated or truncated):
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)

