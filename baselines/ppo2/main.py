import os, sys
lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)


import gym
from baselines.ppo2.ppo2 import learn

def main():
    env = gym.make("HopperPyBulletEnv-v0")
    env.render()
    learn(network = 'mlp',
          env = env,
          total_timesteps = 1e6)



if __name__ == "__main__":
    main()