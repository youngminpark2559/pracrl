# 005_001_lab_Q-learning_on_Nondeterministic_Worlds___05_q_table_frozenlake.py
# q learning with q table using learning rate alpha in non-deterministic environment 

"""
FrozenLake solver using Q-table
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.pjz9g59ap
"""

# You will implement new q learning algorithm,
# which can be used for non-deterministic environment game
# You should listen a little bit of suggestion (q value),
# from Q function (maybe 10% by learning rate)
# And you will keep your claim by 90%

import time
import gym
import numpy as np
import utils.prints as print_utils

N_STATES = 16
N_ACTIONS = 4

LEARNING_RATE = .5
DISCOUNT_RATE = .98

N_EPISODES = 2000

def main():
    """Main"""
    frozone_lake_env = gym.make("FrozenLake-v0")

    # You initialize q table with all zeros
    Q = np.zeros([N_STATES, N_ACTIONS])

    # You set learning parameters

    # You create list to contain summed reward history from episodes
    rewards = []

    for i in range(N_EPISODES):
        # You reset environment and get first new state
        state = frozone_lake_env.reset()
        episode_reward = 0
        done = False

        # Q-Table learning algorithm
        while not done:
            noise = np.random.randn(1, N_ACTIONS) / (i + 1)
            action = np.argmax(Q[state, :] + noise)

            new_state, reward, done, _ = frozone_lake_env.step(action)

            reward = -1 if done and reward < 1 else reward

            # This is new q learning algorithm
            # You will update q table by using learning rate $$$\alpha$$$
            # $$$Q(s,a) \leftarrow (1-\alpha)Q(s,a)+\alpha[r+\gamma max_{a'}Q(s',a')]$$$
            Q[state, action]\
                = (1-LEARNING_RATE)*Q[state,action]+LEARNING_RATE*(reward+DISCOUNT_RATE*np.max(Q[new_state,:]))

            episode_reward += reward
            state = new_state

        rewards.append(episode_reward)

    print("Score over time: " + str(sum(rewards) / N_EPISODES))
    print("Final Q-Table Values")

    for i in range(10):
        # You reset environment and get first new state
        state = frozone_lake_env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = np.argmax(Q[state, :])

            new_state, reward, done, _ = frozone_lake_env.step(action)
            print_utils.clear_screen()
            frozone_lake_env.render()
            time.sleep(.1)

            episode_reward += reward
            state = new_state

            if done:
                print("Episode Reward: {}".format(episode_reward))
                print_utils.print_result(episode_reward)

        rewards.append(episode_reward)

    frozone_lake_env.close()

if __name__ == '__main__':
    main()
