# Part_001.5_Contextual_Bandits.py
# q learning with q network which has multiple states and multiple actions
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class contextual_bandit():
    """
    This class is for definition of arms (action) and slot machine (state)
    """
    def __init__(self):
        # c self.state = 0: means agent sits on first slot machine
        self.state = 0
        
        # There are 3 slot machines
        # Optimal arms are 4th, 2nd, 1st from each slot machine
        # because the less number is, the better arm is
        self.bandits = np.array([[0.2,0,-0.0,-5],[0.1,-5,1,0.25],[-5,5,5,5]])

        # print("self.bandits.shape",self.bandits.shape)
        # c self.bandits.shape (3, 4)


        # c num_bandits: number of slotmachine
        self.num_bandits = self.bandits.shape[0]

        # c num_actions: number of arm in one slotmachine
        self.num_actions = self.bandits.shape[1]

    def getBandit(self):
        """
        This method returns random slot machine (state) like 0  or 1 or 2
        """
        # You randomly choose one slot machine (state)
        # You sit on different slot machine per each episode
        self.state = np.random.randint(0,len(self.bandits)) 
        return self.state

    # c pullArm():     
    def pullArm(self,action):
        """
        This method returns reward (1 or -1) as result of pulling arm
        Args:
            1.action(int):
            index of action
        """
        # c bandit: one specific value determined by state and action,
        # representing specific slot machine (state) and specific arm (action)
        bandit = self.bandits[self.state,action]

        # c result: one random number
        result = np.random.randn(1)

        # If bandit is less than random value,
        # it's good state and action
        if result > bandit:
            return 1
        else:
            return -1

# @
# The Policy-Based Agent

class agent():
    """
    This is agent object,\n
    which suggests action based on given state\n
    Args:
        1.lr:
        1.s_size:
        size of state
        1.a_size:
        size of action
    """
    def __init__(self, lr, s_size,a_size):
        # c state_in in agent class: placeholder for state as input data
        self.state_in= tf.placeholder(shape=[1],dtype=tf.int32)

        # c state_in_OH in agent class: one hot encoded version of state
        state_in_OH = slim.one_hot_encoding(self.state_in,s_size)
        
        # You find value (output) about action based on weight
        output = slim.fully_connected(
            state_in_OH
            ,a_size
            ,biases_initializer=None
            ,activation_fn=tf.nn.sigmoid
            ,weights_initializer=tf.ones_initializer())

        # You reshape output into (4,)
        self.output = tf.reshape(output,[-1])
        # You choose one action by choosing highest value from self.output
        self.chosen_action = tf.argmax(self.output,0)

        # Following 6 lines proceed step of training
        # You send reward and chosen action into network,
        # then, you find loss,
        # then, you update network based on loss
        self.reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
        self.responsible_weight = tf.slice(self.output,self.action_holder,[1])
        # You use cross entropy loss function
        self.loss = -(tf.log(self.responsible_weight)*self.reward_holder)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        self.update = optimizer.minimize(self.loss)

# @
tf.reset_default_graph() 

# c cBandit: contextual_bandit object,
# which has 3 slots whose each slot has 4 bandit
cBandit = contextual_bandit()

# c myAgent: agent object whose state size is 3 and whose action size is 4
myAgent=agent(lr=0.001,s_size=cBandit.num_bandits,a_size=cBandit.num_actions)

# c weights: first trainable variable as weight
weights=tf.trainable_variables()[0] 
total_episodes = 10000 # Entire number of episode when training
# c total_reward: 3by4 array for writing reward history
total_reward = np.zeros([cBandit.num_bandits,cBandit.num_actions]) 
# You define likelihood of randomly choosing actions
e = 0.1 

init = tf.global_variables_initializer()

config = tf.ConfigProto(
        device_count={'GPU': 0}  # uncomment this line to force CPU
    )
    
with tf.Session(config=config) as sess:
    sess.run(init)
    # i stands for current number of episode
    i = 0
    while i < total_episodes:
        # c s: returned one random slot (state) from 3 slots (3 states)
        s = cBandit.getBandit() 

        # e greedy way for selecting action
        # You choose action either from neural network or from random way
        if np.random.rand(1) < e:
            action = np.random.randint(cBandit.num_actions)
        else:
            action = sess.run(myAgent.chosen_action,feed_dict={myAgent.state_in:[s]})
        # You execute chosen action and obtain reward from environment
        reward = cBandit.pullArm(action)

        # You update network
        feed_dict={
            myAgent.reward_holder:[reward]
            ,myAgent.action_holder:[action]
            ,myAgent.state_in:[s]}
        _,ww = sess.run([myAgent.update,weights], feed_dict=feed_dict)
        
        # You record reward
        total_reward[s,action] += reward
        # You print average reward fromm each slot machine per every 500 episode
        if i % 500 == 0:
            print( "Mean reward for each of the " + str(cBandit.num_bandits) + " bandits: " + str(np.mean(total_reward,axis=1)))
        # You increment number of episody by 1
        i+=1

# This prints optimal arm from each slot machine,
# then, evaluates agent does right choice
for a in range(cBandit.num_bandits):
    print("The agent thinks action (arm) " + str(np.argmax(ww[a])+1) + " from slot machine (state) " + str(a+1) + " is most promising.")
    if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
        print("... and agent's action was right!")
    else:
        print("... and agent's action was wrong!")

# Mean reward for each of the 3 bandits: [ 0.   -0.25  0.  ]
# Mean reward for each of the 3 bandits: [39.   42.5  26.25]
# ...
# Mean reward for each of the 3 bandits: [692.25 697.5  619.  ]
# Mean reward for each of the 3 bandits: [731.   732.75 656.  ]

# The agent thinks action (arm) 4 from slot machine (state) 1 is most promising.
# ... and agent's action was right!
# ...
# The agent thinks action (arm) 1 from slot machine (state) 3 is most promising.
# ... and agent's action was right!
