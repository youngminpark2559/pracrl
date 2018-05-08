# 005_001_lab_Q-learning on Nondeterministic Worlds___04_play_frozenlake.py

# @
# This is game in non-deterministic (stochastic) world

import gym
import readchar

import utils.prints as print_utils

# MACROS
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# Key mapping
arrow_keys = {'\x1b[A': UP, '\x1b[B': DOWN, '\x1b[C': RIGHT, '\x1b[D': LEFT}

# Default environment: "is_slippery" True
env = gym.make('FrozenLake-v0')

env.reset()

print_utils.clear_screen()
# Show the initial board
env.render()  

while True:
    # You choose action from keyboard
    key = readchar.readkey()

    if key not in arrow_keys.keys():
        print("Game aborted!")
        break
    
    # You create action by accepting pressed key
    action = arrow_keys[key]

    # You execute action
    state, reward, done, info = env.step(action)

    # You show score board after action
    print_utils.clear_screen()
    env.render()

    print("State: {} Action: {} Reward: {} Info: {}".format(state, action, reward, info))

    if done:
        print_utils.print_result(reward)
