from assets import airspace, airtrafficcontroller
import uam_uav

import gymnasium as gym
import stable_baselines3 as sb3 



env = gym.make('CartPole-v1') #enter string name of uam_env after registering 
observation, info = env.reset()


