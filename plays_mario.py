# -*- coding: utf-8 -*-

# GitHub username: jamdatajam
# Date: 12/06/2024
# Description: A test run for building out AI tools that play videogames to
# find bugs and issues with the game development; this prototype trains an AI
# player to beat Super Mario. In doing so, the goal is to find code issues and
# glitches in the game so that this can be trained and then applied to other
# indie side scroller games. Does so via reinforcement learning.

# installs
pip install gym
pip install gym-super-mario-bros
pip install stable-baselines3
pip install opencv-python-headless

# import gym
import gym_super_mario_bros
from stable_baselines3 import PPO  # Proximal Policy Optimization, a popular RL algorithm

# Create the environment
env = gym.make('SuperMarioBros-v0')

# Initialize the model (a reinforcement learning agent)
model = PPO('CnnPolicy', env, verbose=1)

# Train the agent (for a few steps, usually you'd train longer)
model.learn(total_timesteps=100000)

# Save the trained model
model.save("mario_agent")

# Testing the agent
obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()  # Show the game environment

env.close()
