# -*- coding: utf-8 -*-
"""
Created on Sun May  1 13:15:20 2022

@author: joegr
"""

import gym
from stable_baselines3 import PPO
from snake_env import SnekEnvRendered



env = SnekEnvRendered()
env.reset()

models_dir = "models/1651412934"
model_path = f"{models_dir}/120000.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    env.close()