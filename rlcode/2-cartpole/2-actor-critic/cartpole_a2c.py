# This code uses actor critic architecture to train agent dealing with cartpole environment

import sys
import gym
import pylab
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K

# EPISODES = 1000
EPISODES = 5

class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        
        # You define size of state and size of action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyperparameters for actor critic network
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # You build actor network (policy network)
        self.actor = self.build_actor()
        # You build critic network (value network)
        self.critic = self.build_critic()

        # You build updaters
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()

        if self.load_model:
            self.actor.load_weights("./save_model/cartpole_actor_trained.h5")
            self.critic.load_weights("./save_model/cartpole_critic_trained.h5")

    def build_actor(self):
        """
        This method builds actor network\n
        Actor network takes in state,\n
        and output probability of each action by using softmax in last layer\n
        """
        actor = Sequential()
        actor.add(Dense(24,input_dim=self.state_size,activation='relu',kernel_initializer='he_uniform'))
        actor.add(Dense(self.action_size,activation='softmax',kernel_initializer='he_uniform'))
        print("actor.summary()")                
        actor.summary()
        return actor

    def build_critic(self):
        """
        This method builds critic network\n
        Critic network takes in state,\n
        and output value of each state\n
        """
        critic = Sequential()
        critic.add(Dense(24,input_dim=self.state_size,activation='relu',kernel_initializer='he_uniform'))
        critic.add(Dense(24,input_dim=self.state_size,activation='relu',kernel_initializer='he_uniform'))
        critic.add(Dense(self.value_size,activation='linear',kernel_initializer='he_uniform'))
        print("critic.summary()")
        critic.summary()
        return critic

    def get_action(self, state):
        """
        This method takes in output (state) from actor network (policy network)\n
        Then, this method in probability chooses action\n
        """
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    
    def actor_optimizer(self):
        """
        This method updates actor network (policy network)\n
        """
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob) * advantage
        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights,[],loss)
        train = K.function([self.actor.input,action,advantage],[],updates=updates)

        return train

    def critic_optimizer(self):
        """
        This method updates critic network (value network)\n
        """
        target = K.placeholder(shape=[None, ])
        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)

        return train

    def train_model(self, state, action, reward, next_state, done):
        """
        This method updates actor network and critic network per every time step
        """
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]
        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value

        self.actor_updater([state, act, advantage])
        self.critic_updater([state, target])


if __name__ == "__main__":
    # You will use CartPole-v1 environment
    # Its max time step is 500
    env = gym.make('CartPole-v1')

    # You obtain size of state and size of action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # You instantiate actor critic agent object
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # When episode ends in middle of step, agent gets -100 reward
            reward = reward if not done or score == 499 else -100

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            if done:
                # You print result of training per episode
                score = score if score == 500.0 else score + 100
                scores.append(score)
                episodes.append(e)
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("./save_graph/cartpole_a2c.png")
                plt.plot(episodes, scores, 'b')
                plt.savefig("./save_graph/cartpole_a2c.png")
                print("episode:", e, "  score:", score)
                plt.show()

                # When average score from previous 10 episodes is greater than 490,
                # agent stops training because agent is enough good
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.actor.save_weights("./save_model/cartpole_actor.h5")
                    agent.critic.save_weights("./save_model/cartpole_critic.h5")
                    sys.exit()