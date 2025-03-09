import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv,  DummyVecEnv
import os
import clip
from PIL import Image
import torch
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)
fine_tuned_weights = torch.load('fine_tuned_clip_upper_layers(V6).pth', weights_only = True)
model_clip.load_state_dict(fine_tuned_weights)

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
# model.save("DQN_intersection_new")

#model = PPO.load("ppo_intersection4")
model = DQN.load("DQN_intersection_new")
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
        "initial_vehicle_count": 2,
        'simulation_frequency': 15,
        "spawn_probability": 0.1,
    },
    render_mode = "human"
)

frame_count = 479
save_dir = 'latest_dataset/'

for i in range(20):
        obs, info = env.reset()
        done = False
        truncated = False
        rewards = 0
        time = 0
        proceed = 0
        print(f'episode: {i}  -------------------')
        while not (done or truncated):
            

            # fig, axes = plt.subplots(ncols=1, figsize=(5, 5))
            # axes.imshow(obs[3].T, cmap=plt.get_cmap('gray'))
            # frame_path = os.path.join(save_dir, f"frame_{frame_count}.png")
            # frame_path = os.path.join(save_dir, f"frame_{frame_count}.png")
            # plt.savefig(frame_path)

            
            # with torch.no_grad():
            #     image_features = model_clip.encode_image(preprocessed_image)
            #     text_features = model_clip.encode_text(clip.tokenize(["Slow down.", "Proceed.", "Speed up."]).to(device))
                
            #     logits_per_image = (image_features @ text_features.T).softmax(dim=-1)
            #     predicted_instruction = ["Slow down.", "Proceed.", "Speed up."][logits_per_image.argmax().item()]

                
            # if predicted_instruction == "Slow down.":
            #     action = 0
            # if  predicted_instruction == "Proceed.":    
            #     action = 1
            # if  predicted_instruction == "Speed up.":
            #     action = 2
                      
            action, _states = model.predict(obs)
            # if action == 1:
            #     proceed += 1    
            # if proceed > 3:
            #     print('here')
            #     proceed = 0
            #     action = 2  
            #print(f"Predicted instruction: {predicted_instruction}", "  action:", action, "  frame:", frame_count)
            # print(f"Predicted instruction: {predicted_instruction}", "  action:", action)
            obs, reward, done, truncated, info = env.step(action)

            rewards += reward
            
            # plt.close(fig)
        print("total reward:", rewards)    
  
# for episode in range(5):
#     state = env.reset()
#     terminated = False
#     truncated = False
#     print(episode)
#     while not (terminated or truncated):
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)

