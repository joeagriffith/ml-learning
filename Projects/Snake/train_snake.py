# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:17:50 2022

@author: joegr
"""
import gym
from stable_baselines3 import DQN
from snake_env import SnekEnv
import os
import time

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = SnekEnv()
env.reset()

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
#model_path = f"models/1651412478/120000.zip"
#model = PPO.load(model_path, env=env)

TIMESTEPS = 10000

iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DQN-{int(time.time())}")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")