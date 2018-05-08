# 006_001_lab_Q_Network_for_Frozen_Lake___06_q_net_frozenlake.py
# q network dealing with frozenlake in non-deterministic environment
'''
This code is based on
https://github.com/hunkim/DeepRL-Agents
'''
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# You will run agent in stochastic(non-deterministic) environment
env = gym.make('FrozenLake-v0')

# Input size and output size are predefined based on frozenlake environment
# Dimension of one input vector (one state) will be 16 as one-hot-encoding vector
input_size = env.observation_space.n

# Since you have 4 actions(up, down, left, right),
# Dimension of one output vector (one action) will be 4
output_size = env.action_space.n
learning_rate = 0.1

# These lines establish feed-forward part of network,
# which is used to choose actions
# You will hand in input data (state) by using placeholder X
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)

# Variable node in tensorflow means trainable value
# [1,16][?,?]=[1,4]
# [?,?]=[16,4]
W = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))

# Qpred is q values from $$$\hat{Q}$$$ function, representing predictied actions
Qpred = tf.matmul(X, W)

# Y is output representing one action as one hot vector
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)

# Since it's matrix, you should use reduce_sum
loss = tf.reduce_sum(tf.square(Y - Qpred))
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

dis = .99
num_episodes = 2000

# You create list to save history of summed reward per episode
rList = []

def one_hot(x):
    """
    This method creates array,\n
    to hold one hot encoded vectors (states),\n
    to be used as input data into network,\n
    and this method returns one one hot vector (one state)\n
    """
    # np.identity(16) returns 16by16 array,
    # initialized with value 1 in only diagonal elements,
    # which is useful when you create array holding one hot encoded vectors
    # If state is 0, you need 0 index value from above array,
    # indexing with [0:1] will give you that value
    # If state is 1, you need 1 index value from above array,
    # indexing with [1:2] will give you that value
    # ...
    return np.identity(16)[x:x + 1]

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        s = env.reset()
        e = 1. / ((i / 50) + 10)
        rAll = 0
        done = False
        local_loss = []

        # This is Q-Network training
        while not done:
            # You choose action by greedily 
            # (with e chance of random action) from the Q-network
            # You throw one state as input data
            # You want to obtain predicted q value Qpred (representing action)
            Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})
            if np.random.rand(1) < e:
                # You randomly choose one action
                a = env.action_space.sample()
            else:
                # You choose one action from predicted actions
                a = np.argmax(Qs)

            # You execute obtained action,
            # then, you obtain experience data (reward, new state, ...) from environment
            s1, reward, done, _ = env.step(a)
            # If episode ends,
            if done:
                # you update reward in Q value
                Qs[0, a] = reward
            else:
                # If not done, you update Q with values
                # Obtain the Q_s1 values by feeding new state through our network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
                
                # $$$y=r+\gamma max Q(s')$$$
                # s': next state
                # You will use Qs[0, a](y) as y label in feed_dict={}
                # Update Q
                # You train Q function via only "action a"

                # Reason you use 0 in front of a from Qs[0, a]
                # input * weight = output(action)
                # Let's see shape of them
                # [[1by16]][[16by4]]=[[1by4]]
                # [[1by4]]=[[a1,a2,a3,a4]]
                # Therefore, you use [0,a] index
                Qs[0, a] = reward + dis * np.max(Qs1)

            # You train your network by using target Y and X (state)
            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})

            rAll += reward
            s = s1
        rList.append(rAll)

print("Percent of successful episodes: " + str(sum(rList) / num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
