# 005_001_lab_Q-learning_on_Nondeterministic_Worlds___05_0_q_table_frozenlake.py
# q learning with q table in non-deterministic environment 

# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.pjz9g59ap

# You can use q learning,
# which is supposed to be used for deterministic world, 
# in "non-deterministic environment"
# But it will show more bad result than random action selection

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# You initialize q table with all zeros
# c Q: (16, 4) nparray q table
Q = np.zeros([env.observation_space.n, env.action_space.n])

learning_rate = .85
dis = .99
num_episodes = 2000


rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        # You select one state and all actions,
        # and add noise,
        # Then you use argmax() to select one action
        # c action: chosen one action
        action = np.argmax(Q[state, :] + np.random.randn(1,env.action_space.n) / (i + 1))

        new_state, reward, done, _ = env.step(action)

        # You update Q-Table with reward and q value from next state
        Q[state, action] = reward + dis * np.max(Q[new_state, :])

        # You update state
        state = new_state

        # reward is from each action
        # rAll: summed rewards from all action (from one entire episode)
        rAll += reward

    # You append summed reward per episode as history or summed reward
    rList.append(rAll)

print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
