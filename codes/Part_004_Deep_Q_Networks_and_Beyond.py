# Part_004_Deep_Q_Networks_and_Beyond.py
# q learing with DoubleDQN and DuelingDQN for navigation task

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
# %matplotlib inline
from gridworld import gameEnv

env = gameEnv(partial=False,size=5)

# @
# You build neural network
class Qnetwork():
    """
    This class is for q network object\n
    Args:
        1.h_size:
    """
    def __init__(self,h_size):
        # Network takes in frame as vectorized array from game
        # You resize frame, and you pass resized frame through 4 convolution layers
        
        # Dimension of 21168 is from 84*84*3
        # c scalarInput: nby21168 placeholder for n scalar input data
        self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32)
        
        # You resize 21168 dimension image into 84x84x3 dimension,
        # to be into conv2d()
        # c imageIn: resized image into 84x84x3 to be into conv2d()
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
        
        # Shape of output in case of zero padding 
        # =[{(shape of image)-(size of filter)}/stride]+1
        # (84-8)/4 + 1
        # activation volumn is 20x20x32

        # First convolution layer:
        # 8x8 kernel
        # 4 stride 
        # generating 32 activation maps
        # shape of output is 20x20x32
        # c conv1: image (20x20x32) passed through convolution layer1
        self.conv1 = tf.contrib.layers.convolution2d(
            inputs=self.imageIn
            ,num_outputs=32
            ,kernel_size=[8,8]
            ,stride=[4,4]
            ,padding='VALID'
            ,biases_initializer=None)
        
        # Second convolution layer:
        # 4x4 kernel
        # 2 stride 
        # generating 64 activation maps
        # shape of output is 9x9x64
        # c conv2: image (9x9x64) passed through convolution layer2
        self.conv2 = tf.contrib.layers.convolution2d(
            inputs=self.conv1
            ,num_outputs=64
            ,kernel_size=[4,4]
            ,stride=[2,2]
            ,padding='VALID'
            ,biases_initializer=None)

        # Third convolution layer:
        # 3x3 kernel
        # 1 stride 
        # generating 64 activation maps
        # shape of output is 7x7x64
        # c conv3: node representing image 7x7x64 passed through convolution layer3
        self.conv3 = tf.contrib.layers.convolution2d(
            inputs=self.conv2
            ,num_outputs=64
            ,kernel_size=[3,3]
            ,stride=[1,1]
            ,padding='VALID'
            ,biases_initializer=None)

        # Forth convolution layer:
        # 7x7 kernel
        # 1 stride 
        # generating 512 activation maps
        # shape of output is 1x1x512
        # c self.conv4: node representing image 1x1x512 passed through convolution layer3
        self.conv4 = tf.contrib.layers.convolution2d(
            inputs=self.conv3
            ,num_outputs=512
            ,kernel_size=[7,7]
            ,stride=[1,1]
            ,padding='VALID'
            ,biases_initializer=None)
        
        # You will devide output from last convolution layer,
        # into 2 streams (advantage function, value function)
        # streamAC, streamVC, each is 1by1by256
        # c streamAC: advantage splited from ouput from last convolution layer
        # c streamVC: value splited from ouput from last convolution layer
        self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
        
        # print("self.streamAC",self.streamAC)
        # print("self.streamVC",self.streamVC)
        
        # You vectorize streamAC and streaVC into 256 dimension
        # c self.streamA: vectorized "advantage" splited from ouput from last convolution layer
        # c self.streamV: vectorized "value" splited from ouput from last convolution layer
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        
        # c AW: (256,env.actions) shape node representing weight to be multiplied by streamA to produce advantage
        self.AW = tf.Variable(tf.random_normal([256,env.actions]))
        # c VW: (256,1) shape node representing weight to be multiplied by streamV to produce value
        self.VW = tf.Variable(tf.random_normal([256,1]))
        
        # c self.Advantage: output node from multiplication between vectorized advantage and weight for advantage
        self.Advantage = tf.matmul(self.streamA,self.AW)
        # c self.Value: output node from multiplication between vectorized value and weight for value
        self.Value = tf.matmul(self.streamV,self.VW)
        
        # c self.Qout: predicted q values
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        # c self.predict: predicted action
        self.predict = tf.argmax(self.Qout,1)
        
        # c targetQ: [n] placeholder node for target Q value
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        # c actions: [n] placeholder node for actions
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        # c actions_onehot: I onehotencode "[n] placeholder node for actions"
        self.actions_onehot = tf.one_hot(self.actions,4,dtype=tf.float32)

        # c targetQ: [n] placeholder node for target Q value
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        # c actions: [n] placeholder node for actions
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        # Original code:
        # c actions_onehot: I onehotencode "[n] placeholder node for actions"
        self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype=tf.float32)

        # For those who got gpu error with tf.one_hot()
        # def one_hot_patch(x,depth):
        #     sparse_labels=tf.reshape(x,[-1,1])
        #     derived_size=tf.shape(sparse_labels)[0]
        #     indices=tf.reshape(tf.range(0,derived_size,1),[-1,1])
        #     concated=tf.concat(1,[indices,sparse_labels])
        #     outshape=tf.concat(0,[tf.reshape(derived_size,[1]),tf.reshape(depth,[1])])
        #     return tf.sparse_to_dense(concated, outshape,1.0,0.0)
        # self.actions_onehot = one_hot_patch(self.actions,env.actions)
        
        # You find q value of action from each network
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices=1)
        # 각각의 차이
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        # c trainer: adam optimizer node
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # c self.updateModel: to be trained node
        self.updateModel = self.trainer.minimize(self.loss)

# @
# Experience Replay
class experience_buffer():
    """
    This class stores experience data into buffer memory,\n
    and randomly extracts sample mini batch,\n
    and send them into network\n
    Args:
        1.buffer_size(int):
        default is 50000
    """
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self,experience):
        """
        This method adds experience data into buffer memory\n
        If buffer memory is full,\n
        you delete experience data from olest one (from front one in list),\n
        and you again try to add experience data\n
        Args:
            1.experience:
        """
        if len(self.buffer) + len(experience) >= self.buffer_size:
            # You delete some experience data
            self.buffer[0:(len(experience)+len(self.buffer))-self.buffer_size] = []
        # You append experience data into buffer memory
        self.buffer.extend(experience)
    
    def sample(self,size):
        """
        This method extracts sample mini batch experience data from buffer memory\n
        Args:
            1.size(int):
            size of mini batch
        Returns:
            1.minibatch experience data(nparray):
        """
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])

def processState(states):
    """
    This method reshapes shape of game frame (state) into (21168) shape\n
    Args:
        1.states(nparray):
        game frame image which is store in array
    Returns:
        1.reshaped game frame (state) into (21168) shape
    """
    return np.reshape(states,[21168])

def updateTargetGraph(tfVars,tau):
    """
    This method updates parameters of target network\n
    with parameters of main network\n
    Args:
        1.tfVars(list):
        trainable variables
        1.tau(float):
        trajectory constant,
        representing ratio of updating target network by main network
    Returns:
        1.op_holder(list):
    """
    # c total_vars: number of entire trainable variables
    total_vars = len(tfVars)
    
    op_holder = []
    # 학습 가능한 변수의 절반은 주요 신경망으로, 절반은 타겟 신경망으로
    for idx,var in enumerate(tfVars[0:int(total_vars/2)]):
        # 앞의 절반의 값에 tau 값을 곱하면, 주요 신경망의 weight에 곱해진다.
        # 뒤의 절반의 값에 1-tau 값을 곱하면, 타겟 신경망의 weight에 곱해진다.
        # 아래 코드는 타겟 신경망을 업데이트하는 부분
        op_holder\
            .append(tfVars[int(idx)+int(total_vars/2)]
                .assign((var.value()*tau) + ((1-tau)*tfVars[int(idx)+int(total_vars/2)].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

# @        
# Training the network

batch_size = 32
update_freq = 4 # How ofter will you perform training step
y = .99 # discount factor in respect to target q value
startE = 1
endE = 0.1
anneling_steps = 10000.
num_episodes = 10000
pre_train_steps = 10000 # How many random actions will you use before training?
max_epLength = 50 # max length of one episode, which means max step of one episode
load_model = False # Will you load stored checkpoint?
path = "./dqn" # Directory where you will store checkpoint
# size of output from last convolution layer 
# before it being devided into advantage function and value function
h_size = 512 
tau = 0.001 # ratio of updating target network by main network

tf.reset_default_graph()
# c mainQN: main network object
mainQN = Qnetwork(h_size)
# c targetQN: target network object
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
trainables = tf.trainable_variables()

# c targetOps: values which are used to update target network 
# to make it same with main network
targetOps = updateTargetGraph(trainables,tau)

myBuffer = experience_buffer()

e = startE
stepDrop = (startE - endE)/anneling_steps

jList = []
rList = []
total_steps = 0

# If directory which you configured to save checkpoint doesn't exist,
# you create directory
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    # Will you load checkpoint?
    if load_model == True:
        print ('Loading Model...')
        # You load checkpoint
        ckpt = tf.train.get_checkpoint_state(path)
        # You restore checkpoint
        saver.restore(sess,ckpt.model_checkpoint_path)
    
    sess.run(init)
    # You first synchronize target network by main network
    updateTarget(targetOps,sess) 
    # You start episode
    for i in range(num_episodes):
        # You initialize buffer memory which is used per episode
        episodeBuffer = experience_buffer()
        
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0

        # Q-Network
        # If step exceeds over 50, you terminate episode
        while j < max_epLength:
            j+=1
            # You select action by using decaying e-greedy algorithm

            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,4)
            else:
                # You bring q value by using network
                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
            
            s1,r,d = env.step(a)
            # You reshape shape of state into 21168
            s1 = processState(s1)
            # You increment total step by 1
            total_steps += 1
            # You store experience data into buffer memory
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) 
            
            # If total_steps exceeds over pre_train_steps, you start following code
            if total_steps > pre_train_steps:
                # You decay probability for random selection action
                if e > endE:
                    e -= stepDrop
                
                # you start when satisfying following condition
                if total_steps % (update_freq) == 0:
                    # c trainBatch: randomly extracted mini batch
                    trainBatch = myBuffer.sample(batch_size)
                    # Following code performs double dqn which updates target q value
                    # You select action by using main network
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    # You obtain q values from target network
                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    # You create fake labels in respect to actions
                    end_multiplier = -(trainBatch[:,4] - 1)

                    # You bring values from target q values based on action-th q value in main network
                    # This is core part of double DQN
                    doubleQ = Q2[range(batch_size),Q1]
                    # You add doubleQ multiplied by discount factor and fake labels by instant current reward
                    # In other words, targetQ=instant reward+hightest q value (highest reward) in next state
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    
                    # You update main network with target values
                    # You find loss based on difference,
                    # between target q value and main q value in respect to action
                    _ = sess.run(
                        mainQN.updateModel
                        ,feed_dict={
                            mainQN.scalarInput:np.vstack(trainBatch[:,0])
                            ,mainQN.targetQ:targetQ
                            ,mainQN.actions:trainBatch[:,1]})
                    
                    # You synchronize target network by using values of targetOps from main network
                    # Values of main network will be applied to target network as much as tau
                    updateTarget(targetOps,sess) 
            rAll += r
            s = s1

            if d == True:
                break
        
        # All experience data of current episode will be added into myBuffer
        myBuffer.add(episodeBuffer.buffer)
        # You append entire summed step of current episode into jList
        jList.append(j)
        # You append entire summed reward of current episode into rList
        rList.append(rAll)
        # You save checkpoint periodically
        if i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.cptk')
            print ("Saved Model")
        # You print average reward from last 10 episodes
        if len(rList) % 10 == 0:
            print (total_steps,np.mean(rList[-10:]), e)
    # You save checkpoint
    saver.save(sess,path+'/model-'+str(i)+'.cptk')
# You print success rate
print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

# Saved Model
# 500 2.4 1
# 1000 1.2 1
# ...
# 499500 22.8 0.09999999999985551
# 500000 24.2 0.09999999999985551
# Percent of succesful episodes: 17.7144%
