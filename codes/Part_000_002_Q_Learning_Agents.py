# This code performs q learning with q network

# @
# You will implement q network
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
# %matplotlib inline

env = gym.make('FrozenLake-v0')
# [2017-02-10 11:25:41,941] Making new env: FrozenLake-v0

# You initialize graph
# If you don't do this, in case that you have same nodes, error occurs
tf.reset_default_graph()

# You implement feed forward part of neural network
# Feed forward part is used by agent to select action
# Input data into feed forward is one state represented in 1by16 one hot vector
# c inputs1: 1by16 placeholder node for input data representing one state
inputs1 = tf.placeholder(shape=[1,16],dtype=tf.float32)

# Weight is (16,4)
# input * w=(1,16)(16,4)=(1,4)
# representing 4 actions
# c W: 16by4 placeholder node for weight
W = tf.Variable(tf.random_uniform([16,4],0,0.01))

# c Qout: node of (one state vector)*(weight) representing predicted action (q values)
Qout = tf.matmul(inputs1,W)

# You will find index of highest q value from predicted q values
# c predict: index of hightest predicted q-value
predict = tf.argmax(Qout,1)

# If you input new state into q function, q function ouputs q value representing nextQ
# c nextQ: next Q-value obtained from new state which is obtained from executing action
nextQ = tf.placeholder(shape=[1,4],dtype=tf.float32)

# You find loss by mean squre of difference between (target q (next q))-(predicted q)
loss = tf.reduce_sum(tf.square(nextQ - Qout))

# c trainer: optimizer node
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# You find weight making loss minimum
updateModel = trainer.minimize(loss)

# @
# You will train neural network
# Since everything is operator node in tensorflow,
# before you initialize variables, variables are empty
# c init: initialization node for trainable variable
init = tf.global_variables_initializer()

y = .99
e = 0.1
num_episodes = 20

# Entire step per episode
jList = []
# Entire reward per episode
rList = []

with tf.Session() as sess:
    # If you don't initialize variables, variables are empty
    # So, you need to initialize variables
    sess.run(init)
    
    # You iterate 1 episode, 2 episode, ...
    for i in range(num_episodes):
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        
        # The Q-Network
        while j < 99:
            j+=1
            # You select action based on q value obtained from q network
            # But, you randomly select action by probability e
            # c a: index of highest Q-value
            # c allQ: Q-values obtained from current state
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            # You execute chosen action
            s1,r,d,_ = env.step(a[0])
            
            # You input new state s1 into network,
            # then you get next q value Q' (1,4)
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            
            # You select highest value from q values
            maxQ1 = np.max(Q1)
            
            # You update q table which will be used as target
            targetQ = allQ
            targetQ[0,a[0]] = r + y*maxQ1
            
            # You train network with predicted q value and target q value
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                e = 1./((i/50) + 10)
                break
        # You write end step
        jList.append(j)
        # You write entire reware of episode
        rList.append(rAll)
# You print average reward
print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
# Percent of succesful episodes: 0.13%

# You visualize rList and jList
plt.plot(rList)
plt.show()
plt.plot(jList)
plt.show()
