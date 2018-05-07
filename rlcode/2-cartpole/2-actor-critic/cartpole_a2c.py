# cd /media/young/5e7be152-8ed5-483d-a8e8-b3fecfa221dc/reinforcement-learning-kr-master/2-cartpole/2-actor-critic/
# python cartpole_a2c.py

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


# c A2CAgent class: actor critic agent object
class A2CAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        
        # You define size of state and size of action

        # print("state_size",state_size)
        # c state_size 4
        self.state_size = state_size
        
        # print("action_size",action_size)
        # c action_size 2
        self.action_size = action_size

        self.value_size = 1

        # These are hyperparameters for actor critic network
        self.discount_factor = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.005

        # c self.actor: actor network (policy network) object
        self.actor = self.build_actor()
        # print("self.actor",self.actor)
        # self.actor <keras.models.Sequential object at 0x7f027f2709b0>

        # c self.critic: critic network (value network) object
        self.critic = self.build_critic()
        # print("self.critic",self.critic)
        # c self.critic <keras.models.Sequential object at 0x7fa7e7d0fd68>

        # c actor_updater: updater object for actor network
        self.actor_updater = self.actor_optimizer()
        # print("self.actor_updater",self.actor_updater)
        # c self.actor_updater <keras.backend.tensorflow_backend.Function object at 0x7fdfeb579b00>

        # c self.critic_updater: updater object for critic network
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
        # print("actor",actor)                        
        # actor <keras.models.Sequential object at 0x7f1f90eb19e8>

        actor.add(Dense(self.action_size,activation='softmax',kernel_initializer='he_uniform'))
        # print("actor",actor)                        
        # actor <keras.models.Sequential object at 0x7f4d69ef19e8>

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
        critic.summary()
        return critic

    def get_action(self, state):
        """
        This method takes in output (state) from actor network (policy network)\n
        Then, this method in probability chooses action\n
        """
        # print("state",state)
        # ...
        # state [[ 0.01713044 -0.23849301  0.05034121  0.32082202]]
        # state [[ 0.01236058 -0.04412276  0.05675765  0.04443007]]
        # ...

        # print("self.actor.predict(state, batch_size=1)",self.actor.predict(state, batch_size=1))
        # ...
        # self.actor.predict(state, batch_size=1) [[0.599791   0.40020907]]
        # self.actor.predict(state, batch_size=1) [[0.5717546  0.42824548]]
        # ...

        policy = self.actor.predict(state, batch_size=1).flatten()
        # print("policy",policy)
        # policy [0.5611539 0.4388461]

        # print("np.random.choice(self.action_size, 1, p=policy)[0]",np.random.choice(self.action_size, 1, p=policy)[0])
        # ..
        # np.random.choice(self.action_size, 1, p=policy)[0] 1
        # np.random.choice(self.action_size, 1, p=policy)[0] 1
        # ..

        return np.random.choice(self.action_size, 1, p=policy)[0]

    # c actor_optimizer(): returns "to be trained node" for policy nn
    def actor_optimizer(self):
        """
        This method updates actor network (policy network)\n
        """
        # c action: action placeholder nby2
        action = K.placeholder(shape=[None, self.action_size])
        # print("action",action)
        # action Tensor("Placeholder:0", shape=(?, 2), dtype=float32)

        advantage = K.placeholder(shape=[None, ])

        # print("self.actor.output",self.actor.output)
        # self.actor.output Tensor("dense_2/Softmax:0", shape=(?, 2), dtype=float32)

        action_prob = K.sum(action * self.actor.output, axis=1)
        # print("action_prob",action_prob)
        # action_prob Tensor("Sum:0", shape=(?,), dtype=float32)

        cross_entropy = K.log(action_prob) * advantage

        loss = -K.sum(cross_entropy)

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights,[],loss)
        train = K.function([self.actor.input,action,advantage],[],updates=updates)
        # print("train in actor_optimizer()",train)
        # train in actor_optimizer() <keras.backend.tensorflow_backend.Function object at 0x7f69379bbb00>
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
        # print("train in critic_optimizer()",train)
        # train in critic_optimizer() <keras.backend.tensorflow_backend.Function object at 0x7f6937869240>
        return train

    def train_model(self, state, action, reward, next_state, done):
        """
        This method updates actor network and critic network per every time step
        """
        # print("self.critic.predict(state)",self.critic.predict(state))
        # c value: value from state
        value = self.critic.predict(state)[0]
        # print("value",value)
        # ...
        # value [12.144796]
        # value [13.789681]
        # ...

        # c next_value: next value from next state
        next_value = self.critic.predict(next_state)[0]

        # c act: 1by2 np array for act
        act = np.zeros([1, self.action_size])
        # print("act.shape",act.shape)
        # ...
        # act.shape (1, 2)
        # act.shape (1, 2)
        # ...

        # print("action",action)
        # ...
        # action 0
        # action 1
        # ...

        act[0][action] = 1
        # You find advantage and target by using bellman expectation equation
        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value

        # print("state",state)

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
            # print("action",action)
            # ...
            # action 0
            # action 0
            # action 1
            # ...

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

                # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    agent.actor.save_weights("./save_model/cartpole_actor.h5")
                    agent.critic.save_weights("./save_model/cartpole_critic.h5")
                    sys.exit()
