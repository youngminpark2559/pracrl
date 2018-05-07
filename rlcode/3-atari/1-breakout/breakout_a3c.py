# cd /media/young/5e7be152-8ed5-483d-a8e8-b3fecfa221dc/reinforcement-learning-kr-master/3-atari/1-breakout/
# python breakout_a3c.py

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

# 멀티쓰레딩을 위한 글로벌 변수
global episode
episode = 0
EPISODES = 8000000
# 환경 생성
env_name = "BreakoutDeterministic-v4"


# 브레이크아웃에서의 A3CAgent 클래스(글로벌신경망)
# c A3CAgent: a3c agent object which is global network
class A3CAgent:
    def __init__(self, action_size):
        # 상태크기와 행동크기를 갖고옴
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # print("action_size",action_size)
        # action_size 3

        # A3C 하이퍼파라미터
        self.discount_factor = 0.99
        self.no_op_steps = 30
        self.actor_lr = 2.5e-4
        self.critic_lr = 2.5e-4
        # 쓰레드의 갯수
        self.threads = 8

        # 정책신경망과 가치신경망을 생성
        # c self.actor: actor which is policy network which approximates policy to $$$\pi$$$
        # c self.critic: critic which is value network which approximates value function
        self.actor, self.critic = self.build_model()

        # actor(policy network)과 critic(value network)을 업데이트하는 함수 생성
        # c self.optimizer: list containing self.actor_optimizer(), self.critic_optimizer()
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = \
            tf.summary.FileWriter('summary/breakout_a3c', self.sess.graph)

    # 쓰레드를 만들어 학습을 하는 함수
    # c train(): trains agents on multi threads
    def train(self):
        # 쓰레드 수만큼 Agent 클래스 생성
        # c agents: multiple agents objects as much as number of threads
        agents = [Agent(self.action_size, self.state_size,
                        [self.actor, self.critic], self.sess,
                        self.optimizer, self.discount_factor,
                        [self.summary_op, self.summary_placeholders,
                         self.update_ops, self.summary_writer])
                  for _ in range(self.threads)]

        # 각 쓰레드 시작
        for agent in agents:
            time.sleep(1)
            agent.start()

        # 10분(600초)에 한번씩 모델을 저장
        while True:
            time.sleep(60 * 10)
            self.save_model("./save_model/breakout_a3c")

    # c build_model(): returns policy network (actor), value network (critic)
    def build_model(self):
        input = Input(shape=self.state_size)
        # print("input",input)
        # c input Tensor("input_1:0", shape=(?, 84, 84, 4), dtype=float32)

        conv = Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(input)
        conv = Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(conv)
        
        # c conv: input after convolution layer
        conv = Flatten()(conv)

        # c fc: input after fully connected layer
        fc = Dense(256, activation='relu')(conv)

        # c policy: input after policy function
        policy = Dense(self.action_size, activation='softmax')(fc)

        # c value: input after value function
        value = Dense(1, activation='linear')(fc)
        # print("value",value)
        # value Tensor("dense_3/BiasAdd:0", shape=(?, 1), dtype=float32)

        # c actor: actor network which takes input, outputs policy
        actor = Model(inputs=input, outputs=policy)
        # print("actor",actor)
        # c actor <keras.engine.training.Model object at 0x7fd8d9dacf28>

        # c critic: critic network which takes input, outputs value
        critic = Model(inputs=input, outputs=value)

        # you create function predicting policy (actor)
        actor._make_predict_function()
        # print("actor._make_predict_function()",actor._make_predict_function())
        # actor._make_predict_function() None

        # you create function predicting value (critic)
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        # print("actor",actor)
        # actor <keras.engine.training.Model object at 0x7f3a19b2cf98>

        return actor, critic

    # c actor_optimizer(): updates policy network (actor)
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantages = K.placeholder(shape=[None, ])
        policy = self.actor.output
        # print("policy",policy)
        # c policy Tensor("dense_2/Softmax:0", shape=(?, 3), dtype=float32)

        action_prob = K.sum(action * policy, axis=1)
        # cross entropy loss function about policy
        cross_entropy = K.log(action_prob + 1e-10) * advantages
        cross_entropy = -K.sum(cross_entropy)

        # Remained actors should explore continuously
        # This is entropy loss function for continuous exploration
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        entropy = K.sum(entropy)

        # 두 오류함수를 더해 최종 오류함수를 만듬
        loss = cross_entropy + 0.01 * entropy

        optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
        updates = optimizer.get_updates(self.actor.trainable_weights, [],loss)
        train = K.function([self.actor.input, action, advantages],
                           [loss], updates=updates)
        return train

    # c critic_optimizer(): updates value network (critic)
    def critic_optimizer(self):
        discounted_prediction = K.placeholder(shape=(None,))
        value = self.critic.output

        # [반환값 - 가치]의 제곱을 오류함수로 함
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

    # 각 에피소드 당 학습 정보를 기록
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


# c Agent: running agents object on multiple threads
class Agent(threading.Thread):
    def __init__(self, action_size, state_size, model, sess,
                optimizer, discount_factor, summary_ops):
        
        # You instantiate threading.Thread object
        threading.Thread.__init__(self)

        # A3CAgent 클래스에서 상속
        # You fill these member variables
        self.action_size = action_size
        self.state_size = state_size
        self.actor, self.critic = model
        self.sess = sess
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        [self.summary_op, self.summary_placeholders,
         self.update_ops, self.summary_writer] = summary_ops

        # 지정된 타임스텝동안 experience data을 저장할 리스트
        self.states, self.actions, self.rewards = [], [], []

        # You create loca networks
        self.local_actor, self.local_critic = self.build_local_model()

        self.avg_p_max = 0
        self.avg_loss = 0

        # How often will you update networks?
        self.t_max = 20
        self.t = 0

    def run(self):
        global episode
        # c env: environment object
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

            # 0~30 상태동안 정지
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

                # 1: 정지, 2: 왼쪽, 3: 오른쪽
                if action == 0:
                    real_action = 1
                elif action == 1:
                    real_action = 2
                else:
                    real_action = 3

                # 죽었을 때 시작하기 위해 발사 행동을 함
                if dead:
                    action = 0
                    real_action = 1
                    dead = False

                # 선택한 행동으로 한 스텝을 실행
                next_observe, reward, done, info = env.step(real_action)

                # 각 타임스텝마다 상태 전처리
                next_state = pre_processing(next_observe, observe)
                next_state = np.reshape([next_state], (1, 84, 84, 1))
                next_history = np.append(next_state,history[:, :, :, :3],axis=3)

                # 정책의 최대값
                self.avg_p_max += np.amax(self.actor.predict(
                    np.float32(history / 255.)))

                if start_life > info['ale.lives']:
                    dead = True
                    start_life = info['ale.lives']

                score += reward
                reward = np.clip(reward, -1., 1.)

                # 샘플을 저장
                self.append_sample(history, action, reward)

                if dead:
                    history = np.stack((next_state, next_state,
                                        next_state, next_state), axis=2)
                    history = np.reshape([history], (1, 84, 84, 4))
                else:
                    history = next_history

                # 에피소드가 끝나거나 최대 타임스텝 수에 도달하면 학습을 진행
                if self.t >= self.t_max or done:
                    self.train_model(done)
                    self.update_local_model()
                    self.t = 0

                if done:
                    # 각 에피소드 당 학습 정보를 기록
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

    # k-스텝 prediction 계산
    def discounted_prediction(self, rewards, done):
        discounted_prediction=np.zeros_like(rewards)

        running_add = 0

        if not done:
            running_add = self.critic.predict(np.float32(self.states[-1]/255.))[0]

        for t in reversed(range(0, len(rewards))):
            running_add=running_add*self.discount_factor+rewards[t]
            discounted_prediction[t]=running_add
        return discounted_prediction

    # 정책신경망과 가치신경망을 업데이트
    # c train_model(): trains policy network (actor) and value network (critic)
    def train_model(self, done):
        discounted_prediction = self.discounted_prediction(self.rewards, done)
        # print("discounted_prediction",discounted_prediction)
        # c discounted_prediction [0.26118782 0.26382607 0.26649097 0.2691828  0.27190182 0.27464831
        # 0.27742252 0.28022477 0.28305534 0.28591448 0.2888025  0.2917197
        # 0.29466638 0.2976428  0.30064929 0.30368614 0.30675367 0.30985218
        # 0.31298199 0.31614342]

        # print("discounted_prediction.shape",discounted_prediction.shape)
        # ...
        # discounted_prediction.shape (20,)
        # discounted_prediction.shape (20,)
        # ...

        states = np.zeros((len(self.states), 84, 84, 4))
        # print("states.shape",states.shape)
        # ...
        # states.shape (20, 84, 84, 4)
        # states.shape (20, 84, 84, 4)
        # ...

        # print("len(self.states)",len(self.states))
        # c len(self.states) 20

        # You fill self state into state
        for i in range(len(self.states)):
            states[i] = self.states[i]

        # You normalize state array
        states = np.float32(states/255.)

        # You ask value to critic value network by giving state
        values = self.critic.predict(states)
        values = np.reshape(values, len(values))
        
        # advantages = (q function) - (baseline=value function)
        advantages = discounted_prediction - values

        # You update actor policy network
        self.optimizer[0]([states, self.actions, advantages])
        # You update critic value network
        self.optimizer[1]([states, discounted_prediction])

        self.states, self.actions, self.rewards = [], [], []

    # 로컬신경망을 생성하는 함수
    # c build_local_model(): builds local network for actor network and critic network
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

    # c update_local_model(): updates local network from global network
    def update_local_model(self):
        self.local_actor.set_weights(self.actor.get_weights())
        self.local_critic.set_weights(self.critic.get_weights())

    # policy actor network 에서 출력(policy)을 받아서 확률적으로 행동을 선택
    # c get_action(): returns action index and policy
    def get_action(self, history):
        history=np.float32(history/255.)
        policy = self.local_actor.predict(history)[0]
        action_index = np.random.choice(self.action_size, 1, p=policy)[0]
        return action_index, policy

    # 샘플을 저장
    def append_sample(self, history, action, reward):
        self.states.append(history)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)


# 학습속도를 높이기 위해 흑백화면으로 전처리
# c pre_processing(): processes screen as black and whiite
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
