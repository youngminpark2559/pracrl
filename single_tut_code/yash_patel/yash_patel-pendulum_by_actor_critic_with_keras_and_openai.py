# This code solves pendulum by using actor-critic architecture
# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

import gym
import numpy as np 
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import random
from collections import deque


class ActorCritic:
    """
    This ActorCritic class is for actor critic network object
    """
    def __init__(self, env, sess):
        self.env  = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau   = .125

        # ===================================================================== #
        #                               Actor Model                             #
        # ===================================================================== #
        
        # c self.memory: buffer memory for experience data
        self.memory = deque(maxlen=2000)
        # c self.actor_state_input: state into actor network as input data
        # c self.actor_model: actor network object
        self.actor_state_input, self.actor_model = self.create_actor_model()
        # c self.target_actor_model: target actor network object
        _, self.target_actor_model = self.create_actor_model()

        # where we will feed de/dC (from critic)
        # c self.actor_critic_grad: nby1 node representing gradient for actor-critic network
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]])
        # c actor_model_weights: weight for actor network
        actor_model_weights = self.actor_model.trainable_weights
        # c self.actor_grads: gradient for actor network
        self.actor_grads = tf.gradients(self.actor_model.output,actor_model_weights,-self.actor_critic_grad)
        # c grads: zipped one from self.actor network gradients and actor network weights
        grads = zip(self.actor_grads, actor_model_weights)
        # c self.optimize: node applying grads
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #        

        # c self.critic_state_input: state into critic network as input data
        # c self.critic_action_input: action from actor network into critic network as input data
        # c self.critic_model: critic network object
        self.critic_state_input,self.critic_action_input,self.critic_model=\
            self.create_critic_model()
        # c self.target_critic_model: target critic network object
        _, _, self.target_critic_model=self.create_critic_model()

        # c self.critic_grads: gradient for critic network
        self.critic_grads = tf.gradients(self.critic_model.output,self.critic_action_input) 
        
        # You initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        """
        This method create actor network (policy network)\n
        Returns:
            1.state_input(): state into actor network as input data
            2.model(Model): actor network object
        """
        state_input = Input(shape=self.env.observation_space.shape)

        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.shape[0], activation='relu')(h3)
        
        model = Model(input=state_input, output=output)
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        """
        This method create critic network (value network)\n
        Returns:
            1.state_input(): state into critic network as input data
            1.action_input: action from actor network into critic network as input data
            1.model(Model): critic network object
        """
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)
        
        action_input = Input(shape=self.env.action_space.shape)
        action_h1    = Dense(48)(action_input)
        
        merged    = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model  = Model(input=[state_input,action_input], output=output)
        
        adam  = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        """
        This method stores experience data into buffer memory
        """
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        """
        This method updates actor network (policy network)\n
        Args:
            1.samples: mini batch from buffer memory
        """
        for sample in samples:
            cur_state, action, reward, new_state, _=sample
            predicted_action = self.actor_model.predict(cur_state)

            # c grads: gradient which will be used for update
            grads = self.sess.run(
                self.critic_grads
                ,feed_dict={
                    self.critic_state_input:cur_state
                    ,self.critic_action_input: predicted_action})[0]

            # You update actor network
            self.sess.run(
                self.optimize
                ,feed_dict={
                    self.actor_state_input: cur_state
                    ,self.actor_critic_grad: grads})
            
    def _train_critic(self, samples):
        """
        This method updates critic network (value network)\n
        Args:
            1.samples: mini batch from buffer memory
        """
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)
        
    def train(self):
        """
        This method first call for training,\n
        then this method will call _train_actor() and _train_critic()
        """
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        # c samples: mini batch data from buffer memory
        samples = random.sample(self.memory,batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        """
        This method updates target actor network
        (local actor network) from global actor network
        """
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()
        
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        """
        This method updates target critic network
        (local critic network) from global critic network
        """
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)        

    def update_target(self):
        """
        This method is first call 
        for updating local target actor network and local target critic network
        """
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        """
        This method predicts action from current state,
        by using decaying e-greedy algorithm
        """
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor_model.predict(cur_state)

def main():
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make("Pendulum-v0")
    # c actor_critic: actor critic network object
    actor_critic = ActorCritic(env, sess)

    num_trials = 10000
    trial_len  = 500

    # c cur_state: first state
    cur_state = env.reset()
    # c action: first action
    action = env.action_space.sample()

    while True:
        env.render()
        # c cur_state: reshaped first state to be into network as input data
        cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
        # You predicts action from state by using actor_critic.act()
        action = actor_critic.act(cur_state)
        # c action: reshaped predicted action
        action = action.reshape((1, env.action_space.shape[0]))
        # You execute chosen and reshaped action and get experience data
        new_state, reward, done, _ = env.step(action)
        # You reshape shape of new state
        new_state = new_state.reshape((1, env.observation_space.shape[0]))

        # You save experience data into buffer memory
        actor_critic.remember(cur_state, action, reward, new_state, done)
        # You update actor network and critic network
        actor_critic.train()
        # You convert new state into current state
        cur_state = new_state

if __name__ == "__main__":
    main()
