# This code deals with "partial obserbability markov decision process" question with deep recurrent q network and convolution layer

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import os
import csv
import itertools
import tensorflow.contrib.slim as slim
# %matplotlib inline
from helper import *
from gridworld import gameEnv

env = gameEnv(partial=False,size=9)
env = gameEnv(partial=True,size=9)

# @
# You build neural network
class Qnetwork():
    """
    This class is for q network object\n
    Args:
        1.h_size:
        1.rnn_cell:
        1.myScope:
    """
    def __init__(self,h_size,rnn_cell,myScope):
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
        self.conv1 = slim.convolution2d(
            inputs=self.imageIn
            ,num_outputs=32
            ,kernel_size=[8,8]
            ,stride=[4,4]
            ,padding='VALID'
            ,biases_initializer=None
            ,scope=myScope+'_conv1')
        
        # Second convolution layer:
        # 4x4 kernel
        # 2 stride 
        # generating 64 activation maps
        # shape of output is 9x9x64
        self.conv2 = slim.convolution2d(
            inputs=self.conv1
            ,num_outputs=64
            ,kernel_size=[4,4]
            ,stride=[2,2]
            ,padding='VALID'
            ,biases_initializer=None
            ,scope=myScope+'_conv2')
        
        # Third convolution layer:
        # 3x3 kernel
        # 1 stride 
        # generating 64 activation maps
        # shape of output is 7x7x64
        self.conv3 = slim.convolution2d(
            inputs=self.conv2
            ,num_outputs=64
            ,kernel_size=[3,3]
            ,stride=[1,1]
            ,padding='VALID'
            ,biases_initializer=None
            ,scope=myScope+'_conv3')
        
        # Forth convolution layer:
        # 7x7 kernel
        # 1 stride 
        # generating 512 activation maps
        # shape of output is 1x1x512
        self.conv4 = slim.convolution2d(
            inputs=self.conv3
            ,num_outputs=512
            ,kernel_size=[7,7]
            ,stride=[1,1]
            ,padding='VALID'
            ,biases_initializer=None
            ,scope=myScope+'_conv4')
        
        # c trainLength: how many steps will you use?
        self.trainLength = tf.placeholder(dtype=tf.int32)
        

        # You send output from last convolution layer into recurrent layer
        # You reshape output to be into recurrent layer
        # Reshape should be 
        # [(batch_size)*((trace)*(rnn hidden node unit)]
        # then rnn outputs [batch x units]
        
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])

        # You reshape 1x1x512 into 512
        # You again reshape 512 into [batch_size,trainLength,h_size]
        # c convFlat: reshaped output from last cnn to be into rnn
        self.convFlat = tf.reshape(slim.flatten(self.conv4),[self.batch_size,self.trainLength,h_size])

        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)

        # You use rnn to obtain rnn (output from rnn) and rnn_state (next state)
        self.rnn,self.rnn_state = tf.nn.dynamic_rnn(
            inputs=self.convFlat
            ,cell=rnn_cell
            ,dtype=tf.float32
            ,initial_state=self.state_in
            ,scope=myScope+'_rnn')

        # You vectorize output came from rnn
        self.rnn = tf.reshape(self.rnn,shape=[-1,h_size])
        
        # You half devide output came from rnn 
        # You create 2 streams (value stream and advantage stream)
        # c streamA: stream of advantage
        # c streamV: stream of value
        self.streamA,self.streamV = tf.split(self.rnn,2,1)

        self.AW = tf.Variable(tf.random_normal([int(h_size/2),4]))
        self.VW = tf.Variable(tf.random_normal([int(h_size/2),1]))

        # [n,256][256,4]=[n,4]
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        self.salience = tf.gradients(self.Advantage,self.imageIn)

        # prediction Q value=(value function)+[(advantage function)-(mean of advantage function)]
        self.Qout=\
            self.Value+tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        # You select action based on self.Qout
        # c self.predict: chosen action
        self.predict = tf.argmax(self.Qout,1)
        
        # Mean square error of difference between target q value and prediction q value is loss

        # c targetQ: [n] placeholder node for target Q value
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        # c actions: [n] placeholder node for actions
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        # c actions_onehot: I onehotencode "[n] placeholder node for actions"
        self.actions_onehot = tf.one_hot(self.actions,4,dtype=tf.float32)
        
        # You find q value of action from each network
        self.Q = tf.reduce_sum(tf.multiply(self.Qout,self.actions_onehot), reduction_indices=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        
        # To send precise gradient through network,
        # you will perform masking first half of loss in respect to each history (Lample & Chatlot 2016)
        self.maskA = tf.zeros([self.batch_size,self.trainLength//2]) # number of trace is 4
        self.maskB = tf.ones([self.batch_size,self.trainLength//2])
        self.mask = tf.concat([self.maskA,self.maskB],1)
        self.mask = tf.reshape(self.mask,[-1])

        # You find loss with last half
        self.loss = tf.reduce_mean(self.td_error * self.mask)
        
        # c trainer: adam optimizer node
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        # c self.updateModel: to be trained node
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    """
    This class is for buffer memory object for experience data
    """
    def __init__(self, buffer_size = 1000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    # When you add experience data into buffer list,
    # if buffer list is full,
    # you remove experience data from front
    def add(self,experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1+len(self.buffer))-self.buffer_size] = []
        self.buffer.append(experience)
    
    def sample(self,batch_size,trace_length):
        # c sampled_episodes: You randomly extract sample in batch size from buffer list
        sampled_episodes = random.sample(self.buffer,batch_size)
        sampledTraces = []
        # 이전과 다른 부분, 샘플로 뽑힌 에피소드에서 지정된 크기만큼의 걸음(프레임)을 붙여서 가져온다.
        for episode in sampled_episodes:
            point = np.random.randint(0,len(episode)+1-trace_length)
            sampledTraces.append(episode[point:point+trace_length])
        sampledTraces = np.array(sampledTraces)
        return np.reshape(sampledTraces,[batch_size*trace_length,5])

# @
# You train neural network

batch_size = 4
trace_length = 8
update_freq = 5 # How ofter will you perform training step?
y = .99 # Discount factor for target q value
startE = 1 # Start probability of e for random selecting action
endE = 0.1 # End probability of e for random selecting action
anneling_steps = 10000 # How many steps will you use when decreasing probability from startE to endE
num_episodes = 10000 # How many episodes will you perform?
# num_episodes = 10
# c pre_train_steps: 1000, how many times will you do random action?
pre_train_steps = 10000
load_model = False # Do you want to load checkpoint?
path = "./drqn" # Directory where you save checkpoint
# Size of output from last convolution layer 
# before being devided into value function and advantage function
h_size = 512 
max_epLength = 50 # Max length of episode (50 step)
time_per_step = 1 # git 생성에 사용될 각 걸음의 크기 
summaryLength = 100 # How many episode do you want to save at once?
# summaryLength = 2
tau = 0.001

# You initialize graph
tf.reset_default_graph()

# c cell: LSTM rnn cell for main network
cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
# c cellT: LSTM rnn cell for target network
cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size,state_is_tuple=True)
# c mainQN: main q network object
mainQN = Qnetwork(h_size,cell,'main')
# c targetQN: target q network object
targetQN = Qnetwork(h_size,cellT,'target')

# You initialize variables
init = tf.global_variables_initializer()

# This line prohibits error from being happened
# You uncomment this line to force CPU
config = tf.ConfigProto(device_count={'GPU': 0})

# You instantiate saver object
saver = tf.train.Saver(max_to_keep=5)

# You extract trainable variables
trainables = tf.trainable_variables()

# c targetOps: values which are used to update target network
targetOps = updateTargetGraph(trainables,tau)

# c myBuffer: experience_buffer object
myBuffer = experience_buffer()

# You configure probability e which is used for randomly selecting action 
e = startE
# You decrease probability e as more you proceed episode
stepDrop = (startE - endE)/anneling_steps

# c jList: list which stores entire step per episode
jList = []
# c rList: list which stores entire reward per episode
rList = []
total_steps = 0

# You configure place where you save model
if not os.path.exists(path):
    os.makedirs(path)

# This code is for creating log file which is used for control center
# with open('./Center/log.csv', 'w') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(['Episode','Length','Reward','IMG','LOG','SAL'])    
  
with tf.Session(config=config) as sess:
    # Do you want to load model?
    if load_model == True:
        print ('Loading Model...')
        # You load checkpoint
        ckpt = tf.train.get_checkpoint_state(path)
        # You restore checkpoint
        saver.restore(sess,ckpt.model_checkpoint_path)
    # You initialize variables
    sess.run(init)
    
    # You update target network to sync with main network 
    updateTarget(targetOps,sess)
    
    # You store summary variables for tensorboard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./train',sess.graph)
    
    # You start episode
    for i in range(num_episodes):
        # You create buffer memory as list
        episodeBuffer = []

        # You get initial state
        sP = env.reset()
        # c s: reshaped array into [21168]
        s = processState(sP)
        # done?
        d = False
        # c rAll: summed reward per one episode
        rAll = 0
        # c j: step in episode
        j = 0
        
        # You initialize hidden in recurrent layer
        state = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        
        # Q-Network
        # If step becomes 50, you ends episode
        while j < max_epLength: 
            j+=1
            # This is selecting action by decaying e-greedy algorightm
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                state1 = sess.run(
                    mainQN.rnn_state
                    ,feed_dict={
                        mainQN.scalarInput:[s/255.0]
                        ,mainQN.trainLength:1
                        ,mainQN.state_in:state
                        ,mainQN.batch_size:1})
                # You randomly select action        
                a = np.random.randint(0,4)
            else:
                # You select action (q value) by using network
                a, state1 = sess.run(
                    [mainQN.predict,mainQN.rnn_state]
                    ,feed_dict={
                        mainQN.scalarInput:[s/255.0]
                        ,mainQN.trainLength:1
                        ,mainQN.state_in:state
                        ,mainQN.batch_size:1})
                a = a[0]
            
            # You execute chosen action
            s1P,r,d = env.step(a)
            # You reshape s1P into [21168]
            s1 = processState(s1P)
            # You increment number of step
            total_steps += 1
            # You store experience data into buffer memory 
            episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
            
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq*1000) == 0:
                    print ("Target network updated.")
                    updateTarget(targetOps,sess)

                if total_steps % (update_freq) == 0:
                    # 순환 레이어의 은닉층을 초기화한다.
                    state_train = (np.zeros([batch_size,h_size]),np.zeros([batch_size,h_size])) 
                    
                    # You extract random mini batch data from buffer memory
                    trainBatch = myBuffer.sample(batch_size,trace_length)

                    # You perform double DQN which updates target q value
                    
                    # You select action from main network
                    Q1 = sess.run(
                        mainQN.predict
                        ,feed_dict={
                            mainQN.scalarInput:np.vstack(trainBatch[:,3]/255.0)
                            ,mainQN.trainLength:trace_length
                            ,mainQN.state_in:state_train
                            ,mainQN.batch_size:batch_size})

                    # You obtain q values from target network
                    Q2 = sess.run(
                        targetQN.Qout
                        ,feed_dict={
                            targetQN.scalarInput:np.vstack(trainBatch[:,3]/255.0)
                            ,targetQN.trainLength:trace_length,targetQN.state_in:state_train
                            ,targetQN.batch_size:batch_size})

                    # You create fake label data
                    end_multiplier = -(trainBatch[:,4] - 1)

                    # You bring values from target q values based on action-th q value in main network
                    # This is core part of double DQN
                    doubleQ = Q2[range(batch_size*trace_length),Q1]

                    # 보상에 대한 더블 Q 값을 더해준다. y는 할인 인자
                    # targetQ 는 즉각적인 보상 + 다음 상태의 최대 보상(doubleQ)
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    
                    # You update main network with target values
                    # You find loss based on difference,
                    # between target q value and main q value in respect to action
                    sess.run(
                        mainQN.updateModel
                        ,feed_dict={
                            mainQN.scalarInput:np.vstack(trainBatch[:,0]/255.0)
                            ,mainQN.targetQ:targetQ
                            ,mainQN.actions:trainBatch[:,1]
                            ,mainQN.trainLength:trace_length
                            ,mainQN.state_in:state_train
                            ,mainQN.batch_size:batch_size})
            rAll += r
            s = s1
            sP = s1P
            state = state1

            if d == True:
                break

        # You add episode into buffer memory
        bufferArray = np.array(episodeBuffer)
        episodeBuffer = bufferArray
        myBuffer.add(episodeBuffer)
        jList.append(j)
        rList.append(rAll)

        # You save checkpoint periodically
        if i % 1000 == 0 and i != 0:
            saver.save(sess,path+'/model-'+str(i)+'.cptk')
            print ("Saved Model")
        if len(rList) % summaryLength == 0 and len(rList) != 0:
            print (total_steps,np.mean(rList[-summaryLength:]), e)
#             saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
#                 summaryLength,h_size,sess,mainQN,time_per_step)
    saver.save(sess,path+'/model-'+str(i)+'.cptk')
# Target Set Success
# 5000 0.57 1
# 10000 0.33 1
# Target network updated.
# Target Set Success
# 15000 1.2 0.5499999999998275
# Target network updated.
# Target Set Success
# 20000 2.14 0.09999999999985551
# Target network updated.
# ...
# Target Set Success
# 495000 6.35 0.09999999999985551
# Target network updated.
# Target Set Success
# 500000 5.66 0.09999999999985551

# @
# You will evaluate neural network

e = 0.01
num_episodes = 10000
load_model = True
path = "./drqn"
h_size = 512
max_epLength = 50 
time_per_step = 1
summaryLength = 100

tf.reset_default_graph()

# c cell: LSTM rnn cell for main network
cell = tf.nn.rnn_cell.LSTMCell(num_units=h_size,state_is_tuple=True)
# c cellT: LSTM rnn cell for target network
cellT = tf.nn.rnn_cell.LSTMCell(num_units=h_size,state_is_tuple=True)
# c mainQN: main q network object
mainQN = Qnetwork(h_size,cell,'main')
# c targetQN: target q network object
targetQN = Qnetwork(h_size,cellT,'target')

init = tf.global_variables_initializer()

config = tf.ConfigProto(
        device_count={'GPU': 0}  # uncomment this line to force CPU
    )

saver = tf.train.Saver(max_to_keep=2)

jList = []
rList = []
total_steps = 0

if not os.path.exists(path):
    os.makedirs(path)

##Write the first line of the master log-file for the Control Center
# with open('./Center/log.csv', 'w') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(['Episode','Length','Reward','IMG','LOG','SAL'])    
    
    #wr = csv.writer(open('./Center/log.csv', 'a'), quoting=csv.QUOTE_ALL)
with tf.Session(config=config) as sess:
    if load_model == True:
        print ('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(init)

        
    for i in range(num_episodes):
        episodeBuffer = []
        sP = env.reset()
        s = processState(sP)
        d = False
        rAll = 0
        j = 0
        state = (np.zeros([1,h_size]),np.zeros([1,h_size]))
        
        # The Q-Network
        while j < max_epLength: 
            j+=1
            
            if np.random.rand(1) < e:
                state1 = sess.run(
                    mainQN.rnn_state
                    ,feed_dict={
                        mainQN.scalarInput:[s/255.0]
                        ,mainQN.trainLength:1
                        ,mainQN.state_in:state
                        ,mainQN.batch_size:1})
                a = np.random.randint(0,4)
            else:
                a, state1 = sess.run(
                    [mainQN.predict,mainQN.rnn_state]
                    ,feed_dict={
                        mainQN.scalarInput:[s/255.0]
                        ,mainQN.trainLength:1
                        ,mainQN.state_in:state
                        ,mainQN.batch_size:1})
                a = a[0]
            s1P,r,d = env.step(a)
            s1 = processState(s1P)
            total_steps += 1
            episodeBuffer.append(np.reshape(np.array([s,a,r,s1,d]),[1,5]))
            rAll += r
            s = s1
            sP = s1P
            state = state1
            if d == True:
                break

        bufferArray = np.array(episodeBuffer)
        jList.append(j)
        rList.append(rAll)

        if len(rList) % summaryLength == 0 and len(rList) != 0:
            print (total_steps,np.mean(rList[-summaryLength:]), e)
#             saveToCenter(i,rList,jList,np.reshape(np.array(episodeBuffer),[len(episodeBuffer),5]),\
#                 summaryLength,h_size,sess,mainQN,time_per_step)
print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
# Loading Model...
# 5000 6.53 0.01
# 10000 6.41 0.01
# ...
# 495000 7.13 0.01
# 500000 6.39 0.01
# Percent of succesful episodes: 6.6769%
