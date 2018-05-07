# Part_001_Two_Armed_Bandit.py
# q learning with q network which has one state and multiple actions

import tensorflow as tf
import numpy as np

# @
# You define 1 slot machines (1 states),
# which has 4 arms (4 actions)
# List means goodness of each arm,
# the less number, the better arm
# c bandits: 4 goodness of each arm from 1 slot machine
bandits = [0.2,0,-0.2,-5]
# c num_bandits: number of arm (which is 4) of one slot machine
num_bandits = len(bandits)

def pullBandit(bandit):
    """
    This method generates random numbers\n
    from normal distribution with 0 average data\n
    The less number is, the more reward arm returns\n
    You want agent to select arm which gives highest reward\n
    Args:
        1.bandit(float or int):
        specific one action from one slot machine
    Returns:
        1.reward(int):
        This method returns reward as 1 or -1,
        as result of pulling arm of slot machine
    """
    # c result: one random number,
    # from normal distribution number data having 0 as average
    result = np.random.randn(1)
    
    # If bandit is less than random number, agent obtains 1 reward
    if result > bandit:
        return 1
    else:
        return -1

# @
# The Agent
tf.reset_default_graph()

# Following 2 lines build feed forward part of network
# Feed forward part actually selects action
weights = tf.Variable(tf.ones([num_bandits]))
# print("weights",weights)
# weights <tf.Variable 'Variable:0' shape=(4,) dtype=float32_ref>

# You select index of highest weight,
# and consider that index as index of chosen action
chosen_action = tf.argmax(weights,0)

# Following 6 lines build step of training
# You obtain reward, send chosen action into network, find loss, update network

reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
# You bring weight related to action from entire weight
# c responsible_weight: weight which is related to action, obtained from weights
responsible_weight = tf.slice(weights,action_holder,[1])
# print("responsible_weight",responsible_weight)
# responsible_weight Tensor("Slice:0", shape=(1,), dtype=float32)

# You find loss function with cross entropy loss function
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

# @
# Training the Agent
total_episodes = 1000

# c total_reward: total reward for slot machine
total_reward = np.zeros(num_bandits)
# print("total_reward",total_reward)
# total_reward [0. 0. 0. 0.]

e = 0.1

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    i = 0
    # If current number of episode doesn't exceed 1000, agent keeps going
    while i < total_episodes:
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)
            
        reward = pullBandit(bandits[action])

        # reward_holder: 1 dimensional array
        # reward: integer scalar
        # This part updates network
        # You update weight of network basend on chosen action, future reward,...
        _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
        
        total_reward[action] += reward
        
        if i % 50 == 0:
            print( "Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
        
        i+=1

# You select most promising arm from slot machine
print ("The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising....")

if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print ("...and it was right!")
else:
    print ("...and it was wrong!")
# Running reward for the 4 bandits: [-1.  0.  0.  0.]
# Running reward for the 4 bandits: [  3.  -1.  -1.  34.]
# ...
# Running reward for the 4 bandits: [   2.   -1.    5.  809.]
# Running reward for the 4 bandits: [   4.    0.    4.  853.]

# The agent thinks bandit 4 is the most promising....
# ...and it was right!
