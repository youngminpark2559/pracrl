# Part_003_Model_Based_RL.py
# build model network and policy network reflecting real environment for RL question.html

import numpy as np
import _pickle as pickle
import tensorflow as tf
# %matplotlib inline
import matplotlib.pyplot as plt
import math
import gym
env = gym.make('CartPole-v0')

# @
H = 8
learning_rate = 1e-2
gamma = 0.99
decay_rate = 0.99
resume = False # Will you start from previous checkpoint?

# c model_bs: batch size when you train on model environment
model_bs = 3
# c real_bs: batch size when you train on real environment
real_bs = 3
D = 4 # Dimension of input data

# @
# Policy Network

tf.reset_default_graph()

# c observation: nby4 placeholder node for state as input data
observations = tf.placeholder(tf.float32, [None,4] , name="input_x")
# c W1: 4by8 variable node for weight1 which will be multiplied by input data
W1=tf.get_variable("W1",shape=[4,H],initializer=tf.contrib.layers.xavier_initializer())
# You pass multiplication result through relu activation function
layer1 = tf.nn.relu(tf.matmul(observations,W1))

#  8개의 은닉 노드로 1개의 점수를 낸다
# c W2: 8by1 variable node for weight2 which will be multiplied by output data from layer1
W2 = tf.get_variable("W2", shape=[H, 1],initializer=tf.contrib.layers.xavier_initializer())

# c score: node from multiplication of output from layer1 and weight2, representing score
score = tf.matmul(layer1,W2)

# c probability: node after passing score node through sigmoid function, 
# representing probability of each action
probability = tf.nn.sigmoid(score)

# You collect variables you will train
tvars = tf.trainable_variables()
# c input_y: nby1 placeholder for action's temporal fake label
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
# c advantages: placeholder for advantages
advantages = tf.placeholder(tf.float32,name="reward_signal")

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)


# c W1Grad: placeholder for weight in layer1
W1Grad = tf.placeholder(tf.float32,name="batch_grad1")
# c W2Grad: placeholder for weight in layer2
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
# You will collect gradients before update
# c batchGrad: [weight1 placeholder, weight2 placeholder]
batchGrad = [W1Grad,W2Grad]

# c loglik: loglik node represents how precise of probability in respect to each action
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
# 위에서 구한 loglik와 할인된 보상을 곱해서 손실로 계산
loss = -tf.reduce_mean(loglik * advantages) 
# 그라디언트를 구한다
# c newGrads: gradient from loss and tvars
newGrads = tf.gradients(loss,tvars)
# 그라디언트를 업데이트함
updateGrads = adam.apply_gradients(zip(batchGrad,tvars))


# @
# You will implement model environment network

mH = 256 # number of nodes in hiddel layer

# c input_data: nby5 shape node representing state and information of action
input_data = tf.placeholder(tf.float32, [None, 5])
# c previous_state: nby5 shape node representing previous state
previous_state = tf.placeholder(tf.float32, [None,5] , name="previous_state")
# c W1M: 5by256 shape node representing weight in layer1 of model network
W1M = tf.get_variable("W1M", shape=[5, mH],initializer=tf.contrib.layers.xavier_initializer())
# c B1M: 256 shape node representing bias in layer1 of model network
B1M = tf.Variable(tf.zeros([mH]),name="B1M")
# c layer1M: node representing output from layer1 of model network
layer1M = tf.nn.relu(tf.matmul(previous_state,W1M) + B1M)

# c W2M: 256by256 shape node representing weight in layer2 of model network
W2M = tf.get_variable("W2M", shape=[mH, mH],initializer=tf.contrib.layers.xavier_initializer())
# c B2M: 256 shape node representing bias in layer2 of model network
B2M = tf.Variable(tf.zeros([mH]),name="B2M")
# c layer2M: node representing output from layer2 of model network
layer2M = tf.nn.relu(tf.matmul(layer1M,W2M) + B2M)

# c wO: 256by4 shape node representing weight for state
wO = tf.get_variable("wO", shape=[mH, 4],initializer=tf.contrib.layers.xavier_initializer())
# c wR: 256by1 shape node representing weight for reward
wR = tf.get_variable("wR", shape=[mH, 1],initializer=tf.contrib.layers.xavier_initializer())
# c wD: 256by1 shape node representing weight for done
wD = tf.get_variable("wD", shape=[mH, 1],initializer=tf.contrib.layers.xavier_initializer())

# c bO: 4 shape node representing bias for state
bO = tf.Variable(tf.zeros([4]),name="bO")
# c bR: 1 shape node representing bias for reward
bR = tf.Variable(tf.zeros([1]),name="bR")
# c bD: 1 shape node representing bias for done
bD = tf.Variable(tf.ones([1]),name="bD")

# c predicted_observation: node representing predicted state
predicted_observation = tf.matmul(layer2M,wO,name="predicted_observation") + bO
# c predicted_reward: node representing predicted reward
predicted_reward = tf.matmul(layer2M,wR,name="predicted_reward") + bR
# c predicted_done: node representing predicted done
predicted_done = tf.sigmoid(tf.matmul(layer2M,wD,name="predicted_done") + bD)

# c predicted_observation: node representing target state
true_observation = tf.placeholder(tf.float32,[None,4],name="true_observation")
# c true_reward: node representing target reward
true_reward = tf.placeholder(tf.float32,[None,1],name="true_reward")
# c true_done: node representing target done
true_done = tf.placeholder(tf.float32,[None,1],name="true_done")


# c predicted_state: node representing predicted state
predicted_state = tf.concat([predicted_observation,predicted_reward,predicted_done],1)
# c observation_loss: node representing loss in respect to state
observation_loss = tf.square(true_observation - predicted_observation)

# c reward_loss: loss node in respect to reward
reward_loss = tf.square(true_reward - predicted_reward)
# c done_loss: loss node at ending moment
done_loss = tf.multiply(predicted_done, true_done) + tf.multiply(1-predicted_done, 1-true_done)
done_loss = -tf.log(done_loss)
# c model_loss: loss node combined with reward_loss and done_loss
model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)

modelAdam = tf.train.AdamOptimizer(learning_rate=learning_rate)
# c updateModel: node minimizing model_loss node
updateModel = modelAdam.minimize(model_loss)

# @
# Helper-functions

def resetGradBuffer(gradBuffer):
    """
    This method initializes gradBuffer
    """
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    return gradBuffer
  
def discount_rewards(r):
    """
    This method takes in 1 dimensional float array of rewards,\n
    and computes discounted reward\n
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def stepModel(sess, xs, action):
    """
    This method outputs new state,
    when this method has previous state and action
    """
    # c toFeed: you combine state and action, then, you reshape it into 1by5
    toFeed = np.reshape(np.hstack([xs[-1][0],np.array(action)]),[1,5])
    # c myPredict: You run graph up to predicted_state node, with throwing previous_state (toFeed), to create prediction
    myPredict = sess.run([predicted_state],feed_dict={previous_state: toFeed})
    # c reward: first element from myPredict, select all rows, select 4th column, representing predicted reward
    reward = myPredict[0][:,4]
    # c observation: predicted states
    observation = myPredict[0][:,0:4]
    # You adjust value of state
    observation[:,0] = np.clip(observation[:,0],-2.4,2.4)
    observation[:,2] = np.clip(observation[:,2],-0.4,0.4)
    
    # You evaluate done or not
    doneP = np.clip(myPredict[0][:,5],0,1)
    
    # If prediction of state at ending moment is greater than 0.1,
    # or if number of states is greater than 300,
    # you consider as you've finished
    if doneP > 0.1 or len(xs)>= 300:
        done = True
    else:
        done = False
    return observation, reward, done

# @
# You will train policy and model environment network
xs,drs,ys,ds = [],[],[],[]

running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1

init = tf.global_variables_initializer()

# c batch_size: at initial, batch size is from real_bs
batch_size = real_bs
# c drawFromModel: will you use model environment about state?
drawFromModel = False

# c trainTheModel: will you train model environment?
trainTheModel = True
# c trainThePolicy: will you train policy? at initial, you don't train policy
trainThePolicy = False
switch_point = 1

with tf.Session() as sess:
    rendering = False

    sess.run(init)

    observation = env.reset()
    x = observation
    
    # You run graph up to tvars, and get gradBuffer
    gradBuffer = sess.run(tvars)
    # You initialize gradBuffer
    gradBuffer = resetGradBuffer(gradBuffer)
    
    # You iterate 5000 episodes
    # while episode_number <= 5000:
    while episode_number <= 10:
        # After showing good enough performance, then you display activity on screen
        if (reward_sum/batch_size > 150 and drawFromModel == False) or rendering == True : 
            env.render()
            rendering = True
        
        # You reshape shape of state
        x = np.reshape(observation,[1,4])
        
        # You create probability values in respect to actions
        tfprob = sess.run(probability,feed_dict={observations: x})
        
        # If tfprob is less than random number, you do action 1
        # If tfprob is greater than random number, you do action 0
        action = 1 if np.random.uniform() < tfprob else 0

        # You append each x (state) into xs (state list),
        # not to lose x (state) when backpropagation
        xs.append(x) 
        
        # You create fake label data paired with actions
        y = 1 if action == 0 else 0 
        ys.append(y)
        
        # You decide whether you will use real environment or model environment
        # Then, you display next state, reward, done or not
        if drawFromModel == False:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done = stepModel(sess,xs,action)
        
        reward_sum += reward
        # You append done cases into ds list
        ds.append(done*1)
        # # You append rewards into drs list
        drs.append(reward)

        # If episode finishes,
        if done: 
            # if you didn't bring from model envrioment,
            # you increment real_episodes by 1
            if drawFromModel == False: 
                real_episodes += 1
            # You increment episode_number by 1
            episode_number += 1

            # For training, you arrange state, reward, done, label
            # c epx: stacked states
            epx = np.vstack(xs)
            # c epy: stacked fake lables about actions
            epy = np.vstack(ys)
            # c epr: stacked rewards
            epr = np.vstack(drs)
            # c epr: stacked dones
            epd = np.vstack(ds)

            # You flush each list
            xs,drs,ys,ds = [],[],[],[]
            
            # If you decided to train model envrioment,
            if trainTheModel == True:
                # You extract target actions from fake labels list
                actions = np.array([np.abs(y-1) for y in epy][:-1])
                state_prevs = epx[:-1,:]
                # You conbine previous state and action, with creating 5 dimension
                state_prevs = np.hstack([state_prevs,actions])
                
                # You check next state
                state_nexts = epx[1:,:]
                # You check reward
                rewards = np.array(epr[1:,:])
                # You check done
                dones = np.array(epd[1:,:])
                
                # To train model environment network, you combine experience data (which is actually meaningless)
                state_nextsAll = np.hstack([state_nexts,rewards,dones])

                # You will throw previous state, target state, target done, target reward
                feed_dict={
                    previous_state:state_prevs
                    ,true_observation:state_nexts
                    ,true_done:dones
                    ,true_reward:rewards}

                # You run graph up to model_loss node, predicted_state node, updateModel node,
                # with throwing feed_dict,
                # then, you obtain loss and pState (predicted next state)
                loss,pState,_ = sess.run([model_loss,predicted_state,updateModel],feed_dict)

            # If you decided to train policy network
            if trainThePolicy == True:
                discounted_epr = discount_rewards(epr).astype('float32')
                # You normalize discounted rewards
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                
                # You run graph up to newGrads node,
                # with throwing stacked states, target labels, discount_rewards,
                # then, you get tGrad
                tGrad = sess.run(
                    newGrads
                    ,feed_dict={
                        observations:epx
                        ,input_y:epy
                        ,advantages:discounted_epr})
                
                # If tGrad is too large, you terminate training
                if np.sum(tGrad[0] == tGrad[0]) == 0:
                    break
                
                # You append gradients into gradBuffer
                for ix,grad in enumerate(tGrad):
                    gradBuffer[ix] += grad
            
            # If you've performed enough episodes,
            if switch_point + batch_size == episode_number: 
                switch_point = episode_number
                # you apply collected gradients into network to update policy network
                if trainThePolicy == True:
                    sess.run(updateGrads,feed_dict={W1Grad:gradBuffer[0],W2Grad:gradBuffer[1]})
                    gradBuffer = resetGradBuffer(gradBuffer)

                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                
                # If you don't use state from model environment,
                if drawFromModel == False:
                    print ('World Perf: Episode %f. Reward %f. action: %f. mean reward %f.' % (real_episodes,reward_sum/real_bs,action, running_reward/real_bs))
                    # you terminate when over 200
                    if reward_sum/batch_size > 200:
                        break
                reward_sum = 0

                # After training 100 episodes, you train policy from model environment
                if episode_number > 100:
                    drawFromModel = not drawFromModel
                    trainTheModel = not trainTheModel
                    trainThePolicy = not trainThePolicy
            
            # If you use state from model environment
            if drawFromModel == True:
                # first, you initialize model randomly
                observation = np.random.uniform(-0.1,0.1,[4])
                batch_size = model_bs
            # or if you don't use state from model environment,
            # you can use env.reset()
            else:
                observation = env.reset()
                batch_size = real_bs
                
# print (real_episodes)
# World Perf: Episode 4.000000. Reward 21.666667. action: 0.000000. mean reward 21.666667.
# World Perf: Episode 7.000000. Reward 35.000000. action: 1.000000. mean reward 21.800000.
# ...
# World Perf: Episode 535.000000. Reward 76.000000. action: 1.000000. mean reward 676521155727795368403217766416384.000000.
# World Perf: Episode 538.000000. Reward 246.000000. action: 0.000000. mean reward 663058403058061947241155428089856.000000.

plt.figure(figsize=(8, 12))

for i in range(6):
    plt.subplot(6, 2, 2*i + 1)
    plt.plot(pState[:,i])
    plt.subplot(6,2,2*i+1)
    plt.plot(state_nextsAll[:,i])
plt.tight_layout()
plt.show()
# img 2018-05-03 19-27-42.png
# </xmp><img src="https://raw.githubusercontent.com/youngmtool/pracrl/master/codes/pic/2018-05-03 19-27-42.png"><xmp>
