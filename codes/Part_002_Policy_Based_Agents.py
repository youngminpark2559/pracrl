# Part_002_Policy_Based_Agents.py
# q learning with q network where you use policy gradient based agent in cartpole question

import numpy as np
# python 3.5
import _pickle as pickle 
import tensorflow as tf
# %matplotlib inline
import matplotlib.pyplot as plt
import math
import gym

# @
env = gym.make('CartPole-v0')

env.reset()
random_episodes = 0
reward_sum = 0

# You will try episode 10 times with random actions
# while random_episodes < 10:
while random_episodes < 5:
    env.render()
    # You pass "action" into env.step(),
    # then, you obtain observation (new state), reward, done
    observation, reward, done, _ = env.step(np.random.randint(0,2))
    # You write sum of reward
    reward_sum += reward
    # If pole falls, episode ends and done becomes True
    if done:
        random_episodes += 1
        print ("Reward for this episode was:",reward_sum)
        reward_sum = 0
        env.reset()
# You can see pole falls in very several initial tries,
# (for example, 14 means you holded pole by 14 tries and then pole fell)
# when you try this with random action
# Reward for this episode was: 14.0
# Reward for this episode was: 15.0
# ..
# Reward for this episode was: 30.0
# Reward for this episode was: 42.0
# In conclusion, above result shows bad performance

# @
# You will try episode with gradient based agent

H = 10 # number of nodes in hidden layer
batch_size = 5
learning_rate = 1e-2
gamma = 0.99
D = 4 # dimension of input data (state)

tf.reset_default_graph()

# c observations: [n,D] placeholder node for observation (state)
observations = tf.placeholder(tf.float32, [None,D] , name="input_x")
# c W1: [input data dimension, number of node in layer] variable node in layer1
W1 = tf.get_variable("W1",shape=[D, H],initializer=tf.contrib.layers.xavier_initializer())
# output after relu in layer1
layer1 = tf.nn.relu(tf.matmul(observations,W1))

W2 = tf.get_variable("W2",shape=[H, 1],initializer=tf.contrib.layers.xavier_initializer())
# c score: last value from layer2
score = tf.matmul(layer1,W2)
# c probability: probability node after sigmoid on "last value from layer2"
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32,[None,1], name="input_y")
advantages = tf.placeholder(tf.float32,name="reward_signal")
loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss,tvars)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate) # 최적화기 adam
W1Grad = tf.placeholder(tf.float32,name="batch_grad1") # 그라디언트 저장하는 부분
W2Grad = tf.placeholder(tf.float32,name="batch_grad2")
batchGrad = [W1Grad,W2Grad]

updateGrads = adam.apply_gradients(zip(batchGrad,tvars))

def discount_rewards(r):
    """
    This method takes in array of reward,\n
    then, calculates discounted reward\n
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# c xs: collection of actions in list
# c ys: collection of label related to each action
# c drs: collection of reward in list
xs,drs,ys = [],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
# total_episodes = 10000
total_episodes = 10

init = tf.global_variables_initializer()

with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset()

    # You will collect gradients until you update policy network
    gradBuffer = sess.run(tvars)
    # You initialize gradBuffer
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    
    # You start episode
    while episode_number <= total_episodes:
        
        # Rendering agent's action slows things down, 
        # so, you won't display agent's action until agent works well after enough training
        
        if reward_sum/batch_size > 100 or rendering == True : 
            env.render()
            rendering = True

        # You reshape shape of state to shape which network can deal with            
        x = np.reshape(observation,[1,D])

        # You execute graph up to probability node,
        # with throwing state,
        # then, you obtain probability of actions
        # c tfprob: probability of each action from state ran by neural network
        tfprob = sess.run(probability,feed_dict={observations: x})

        # If probability of action is less than random value,
        # you choose action as 1,
        # or else as 0
        action = 1 if np.random.uniform() < tfprob else 0

        # You append each x (each action) into xs list (collection of actions)
        xs.append(x)
        If action is 0, you create fake label y (in respect to action) as 1,
        or if action is 1, you create fake label y (in respect to action) as 0
        y = 1 if action == 0 else 0
        # You append each y (each label) into ys list (collection of labels)
        ys.append(y)
        
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        
        # Since you lose reward of previous actions once you call step(),
        # you should record reward
        drs.append(reward) 
        
        # When episode finishes,
        if done: 
            episode_number += 1

            # you prepare to update network with state, label, reward of each episode

            # c epx: vertically stacked action list
            epx = np.vstack(xs)

            # c epy: vertically stacked action's label list
            epy = np.vstack(ys)
            
            # c epr: vertically stacked reward list
            epr = np.vstack(drs)
            
            # You initialize them for next episode
            xs,drs,ys = [],[],[] 

            # c discounted_epr: discounted reward
            discounted_epr = discount_rewards(epr)
            
            # You perform set of discounted reward 
            # to have normal distribution having 0 mean and 1 variance
            # to manage variance of gradient

            # You subtract all discounted rewards by mean value of discounted rewards
            discounted_epr -= np.mean(discounted_epr)
            # You devide above discounted rewards by std value of above discounted rewards
            discounted_epr /= np.std(discounted_epr)
            
            # You find gradient of each episode
            # You run graph up to newGrads node,
            # with throwing epx,epy,discount_rewards
            # c tGrad: 2 gradients for 2 weights
            tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})

            # You store obtained gradients into gradient buffer memory gradBuffer
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # When you've performed enough episodes (batch size),
            # you apply gradients which are stored in gradBuffer into network
            if episode_number % batch_size == 0: 
                sess.run(updateGrads,feed_dict={W1Grad:gradBuffer[0],W2Grad:gradBuffer[1]})
                # After you use existing gradients, you empty gradBuffer
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix]=grad*0
                
                # You make statistical result showing how network works well per episode
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print ('Average reward for episode %f.  Total average reward %f.' % (reward_sum/batch_size, running_reward/batch_size))

                # You stop when average reward per episode exceeds 200                
                # if reward_sum/batch_size > 200: 
                if reward_sum/batch_size > 20: 
                    print ("Task solved in",episode_number,'episodes!')
                    break
                
                # You initialize reward sum list
                reward_sum = 0
            # You initialize environment
            observation = env.reset()
        
print (episode_number,'Episodes completed.')
# Average reward for episode 20.000000.  Total average reward 20.000000.
# Average reward for episode 17.400000.  Total average reward 19.974000.
# ...
# Average reward for episode 120.600000.  Total average reward 54.104344.
# Average reward for episode 229.400000.  Total average reward 55.857300.
# Task solved in 730 episodes!
# 730 Episodes completed.
