import ast
import argparse
import logging

import os
import numpy as np
import pdb
# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper,SteeringToWheelVelWrapper


def _enjoy():          
    # Launch the env with our helper function
    #env = launch_env("MultiMap-v0")
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    env = SteeringToWheelVelWrapper(env)
    print("Initialized Wrappers")

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])   
    max_action = float(0.75) # vel and angel limit to 0.8.
    print("ddpg param")
    print(state_dim)
    print(action_dim)
    print(max_action)
    # Initialize policy
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    policy.load(filename='ddpg', directory='reinforcement/pytorch/models/map2_policy')

    obs = env.reset()
    done = False
    actions = []
    T = True

    while T == True:
        while not done:
            action = policy.predict(np.array(obs))
            # Perform action
            #action[0]=0
            #action[1]=0
            actions.append(action)

            obs, reward, done, _ = env.step(action)
            #print("reward is {}".format(reward))
            env.render()
        done = False
        obs = env.reset()
        np.savetxt('./control_v_a_map2_seed16.txt', actions, delimiter=',')
        T = False



def wheel2velangle (action):
    gain=1.0
    trim=0.0
    radius=0.0318
    k=27.0
    limit=1.0
    wheel_distance=0.102
    baseline = wheel_distance

    k_r_inv = (gain+trim)/k
    k_l_inv = (gain+trim)/k

    u_r = action[0]
    u_l = action[1]

    omega_r = u_r/k_r_inv
    omega_l = u_l/k_l_inv

    vel = (omega_r+omega_l)*radius/2
    angle = (omega_r-omega_l)*radius/baseline

    action_vel_angle =  np.array([vel, angle])

    return action_vel_angle


if __name__ == '__main__':
    _enjoy()





       