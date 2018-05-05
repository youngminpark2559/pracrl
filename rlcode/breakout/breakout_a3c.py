# This code trains "breakout atari game" by using A3C architecture.

from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Dense, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import numpy as np
import threading
import random
import time
import gym

# This is global variable for controling multi-threading
global episode
# First, you initialize episode by 0
episode = 0
# You will iterate 8000000 times for training
EPISODES = 8000000
# You create environment
env_name = "BreakoutDeterministic-v4"

# This class is a3c agent object which will be used as global network
class A3CAgent:
    def __init__(self, action_size):
        # You define state size
        self.state_size = (84, 84, 4)
        # You define action size
        self.action_size = action_size

        # You define hyper parameters for A3C global network
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4
        # This is number of threads you will use 
        self.threads = 8

        # self.actor is policy network which approximates policy to policy function $$$\pi$$$
        # self.critic is value network which approximates value to value function
        self.actor, self.critic = self.build_model()

        # self.optimizer is list containing self.actor_optimizer(), self.critic_optimizer(),
        # that each function updates actor (policy network) and critic (value network)
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        # This is configuration for tensorboard
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/breakout_a3c', self.sess.graph)

    # train() trains agents (actor network+critic network) on multi threads
    def train(self):
        # agents is multiple agents objects instantiated as much as number of threads
        agents = [Agent(self.action_size, self.state_size,
                        [self.actor, self.critic], self.sess,
                        self.optimizer, self.discount_factor,
                        [self.summary_op, self.summary_placeholders,
                         self.update_ops, self.summary_writer])
                  for _ in range(self.threads)]

        # You start each thread
        for agent in agents:
            time.sleep(1)
            agent.start()

        # You save model every 10 minutes
        while True:
            time.sleep(60 * 10)
            self.save_model("./save_model/breakout_a3c")

    # build_model() returns policy network (actor) and value network (critic)
    def build_model(self):
        input = Input(shape=self.state_size)

        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        
        # conv is convolution layer
        conv = Flatten()(conv)

        # fc is fully connected layer
        fc = Dense(256, activation='relu')(conv)

        # policy will be used as policy function
        policy = Dense(self.action_size, activation='softmax')(fc)
        
        # value will be used as value function
        value = Dense(1, activation='linear')(fc)

        # This is actor network which takes input, outputs policy
        actor = Model(inputs=input, outputs=policy)

        # This is critic network which takes input, outputs value
        critic = Model(inputs=input, outputs=value)

        # You create function predicting policy (actor)
        actor._make_predict_function()

        # You create function predicting value (critic)
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    # actor_optimizer() updates policy network (actor)
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])
        policy = self.actor.output

        action_prob = K.sum(action * policy, axis=1)
        # Cross entropy loss function about policy
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # Remained actors should continuously explore environment
        # This is entropy loss function for continuous exploration
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # You create final loss function by adding two loss functions
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    # critic_optimizer() updates value network (critic)
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))
        value = self.critic.output

        # You use loss function as mean squre error
        loss = K.mean(K.square(discounted_prediction - value))

        optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.critic.trainable_weights, [],loss)
        train = K.function([self.critic.input, discounted_prediction],
                           [loss], updates=updates)
        return train

    def load_model(self, name):
        self.actor.load_weights(name + "_actor.h5")
        self.critic.load_weights(name + "_critic.h5")

    def save_model(self, name):
        self.actor.save_weights(name + "_actor.h5")
        self.critic.save_weights(name + "_critic.h5")

    # You record each episode's training history
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Prob/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)

        summary_vars = [episode_total_reward,
                        episode_avg_max_q,
                        episode_duration]

        summary_placeholders = [tf.placeholder(tf.float32)
                                for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i])
                      for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


# Agent class is for running agents object on multiple threads
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess,
                optimizer, discount_factor, summary_ops):
        
        # You instantiate parent threading.Thread object
        threading.Thread.__init__(self)

        # You fill these member variables
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,
         self.update_ops, self.summary_writer] = summary_ops

        # These lists are needed to store experience data
        self.states, self.actions, self.rewards = [], [], []

        # You create local networks (local actor network, local critic network)
        self.local_actor, self.local_critic = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0

        # How often will you update networks?
        self.t_max = 20
        self.t = 0

    def run(self):
        global episode
        # env is environment object
        env = gym.make(env_name)

        step = 0

        while episode < EPISODES:
            # You initialize done and dead
            done = False
            dead = False

            # You initialize score and start_list
            score, start_life = 0, 5

            observe = env.reset()
            next_observe = observe

            for _ in range(random.randint(1, 30)):
                observe = next_observe
                next_observe, _, _, _ = env.step(1)

            state = pre_processing(next_observe, observe)
            history = np.stack((state, state, state, state), axis=2)
            history = np.reshape([history], (1, 84, 84, 4))

            while not done:
                step += 1
                self.t += 1
                observe = next_observe
                action, policy = self.get_action(history)

                # 1: stop, 2: left, 3: right
                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3

                # When agent dies, this code makes restart with firing missle
                if dead:
                    action = 0
                    real_action = 1
                    dead = False

                # You execute one step by chosen action
                next_observe, reward, done, info = env.step(real_action)

                # You perform pre processing (which converts screen black and white) every each time step
                next_state = pre_processing(next_observe, observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state,history[:, :, :, :3],axis=3)

                # This is max value of policy
                self.avg_p_max += np.amax(self.actor.predict(
                    np.float32(history / 255.)))

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                # You save sample
                self.append_sample(history, action, reward)

                if dead:
                    history = np.stack((next_state, next_state,
                                        next_state, next_state), axis=2)
                    history = np.reshape([history], (1, 84, 84, 4))
                else:
                    history = next_history

                # If epidode ends or it reaches to max time step,
                # you start training
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_local_model()
                    self.t = 0

                if done:
                    episode += 1
                    print("episode:",episode," score:",score," step:",step)

                    stats = [score,self.avg_p_max/float(step),step]
                    for i in range(len(stats)):
                        self.sess.run(self.update_ops[i], feed_dict={
                            self.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = self.sess.run(self.summary_op)
                    self.summary_writer.add_summary(summary_str, episode + 1)
                    self.avg_p_max = 0
                    self.avg_loss = 0
                    step = 0

    def discounted_prediction(self, rewards, done):
        discounted_prediction=np.zeros_like(rewards)

        running_add = 0

        if not done:
            running_add = self.critic.predict(np.float32(self.states[-1]/255.))[0]

        for t in reversed(range(0, len(rewards))):
            running_add=running_add*self.discount_factor+rewards[t]
            discounted_prediction[t]=running_add
        return discounted_prediction

    # train_model() trains policy network (actor) and value network (critic)
    def train_model(self, done):
        discounted_prediction = self.discounted_prediction(self.rewards, done)

        states = np.zeros((len(self.states), 84, 84, 4))
        
        # You fill "self state" into state
        for i in range(len(self.states)):
            states[i] = self.states[i]

        # You normalize state array
        states = np.float32(states/255.)

        # You ask value to "critic value network" by giving state to critic network
        values = self.critic.predict(states)
        values = np.reshape(values, len(values))
        
        # advantages = (q function) - (baseline=value function)
        advantages = discounted_prediction - values

        # You update actor policy network
        self.optimizer[0]([states, self.actions, advantages])

        # You update critic value network
        self.optimizer[1]([states, discounted_prediction])

        self.states, self.actions, self.rewards = [], [], []

    # build_local_model() builds local networks (actor network and critic network)
    def build_local_model(self):
        input = Input(shape=self.state_size)

        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        conv = Flatten()(conv)
        fc = Dense(256, activation='relu')(conv)

        policy = Dense(self.action_size, activation='softmax')(fc)
        value = Dense(1, activation='linear')(fc)

        local_actor = Model(inputs=input, outputs=policy)
        local_critic = Model(inputs=input, outputs=value)

        local_actor._make_predict_function()
        local_critic._make_predict_function()

        local_actor.set_weights(self.actor.get_weights())
        local_critic.set_weights(self.critic.get_weights())

        local_actor.summary()
        local_critic.summary()

        return local_actor, local_critic

    # update_local_model() updates local networks by global network's parameters
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    # get_action() returns action's index and policy
    def get_action(self, history):
        history=np.float32(history/255.)
        policy = self.local_actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # You save sample
    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


# c pre_processing(): processes screen as black and whiite to increase training speed
def pre_processing(next_observe, observe):
    processed_observe = np.maximum(next_observe, observe)
    processed_observe = np.uint8(
        resize(rgb2gray(processed_observe)
        ,(84, 84)
        ,mode='constant')*255)
    return processed_observe

if __name__ == "__main__":
    global_agent = A3CAgent(action_size=3)
    global_agent.train()
