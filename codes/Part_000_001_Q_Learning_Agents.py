# This code performs q learning with q table

# Q-Table Learning
import gym
import numpy as np

# You load frozenlake environment
# c env: frozen lake environment object
env = gym.make('FrozenLake-v0')

# @
# You will implment q learning algorithm by using q table

# You initialize all squares by 0 representing policy in q table
# All squares representing policy are generated from state and action
# env.observation_space.n: 16 states
# env.action_space.n: 4 actions
# c Q: 16by4 q table which is composed of state(row) and action(column)
Q = np.zeros([env.observation_space.n,env.action_space.n])

lr = .85
# c y: discount factor for future reward
y = .99
# num_episodes = 2000
num_episodes = 5

# c jList: list which stores entire step per episode
jList = []
# c rList: list which stores entire reward per episode
rList = []

# You start episode
for i in range(num_episodes):
    # You create buffer memory as list
    episodeBuffer = []
    
    sP = env.reset()
    # c s: reshaped array into [21168]
    s = processState(sP)
    # done?
    d = False
    # c rAll: summed reward per one episode
    rAll = 0
    # c j: step in episode
    j = 0

for i in range(num_episodes):
    # You get initial state
    # c s: initial state
    s = env.reset()
    # c rAll: summed reward per one episode
    rAll = 0
    # done?
    d = False
    # c j: step in episode
    j = 0

    # Q-Network
    # If step exceeds 99, you ends episode
    while j < 99:
        # You increment 1 step
        j+=1
        # Q table에서 e -greedy 에 따라 가장 좋은 행동을 선택함
        # 매 걸음마다 랜덤적 요소를 넣음
        # 1/(i+1) 을 넣는 이유는 에피소드가 진행될 수록 랜덤적 요소를 줄이려고 하는 것임
        # c a: chosen action
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        # You execute chosen action
        s1,r,d,_ = env.step(a)
        # You update previous q table with new experience data
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        # 에피소드 총 보상에서 더해줌
        rAll += r
        s = s1
        # If you reach to end, you go over to next episode
        if d == True:
            break
    jList.append(j)
    # 에피소드별 총 보상을 모음
    rList.append(rAll)
print ("Score over time: " +  str(sum(rList)/num_episodes))
# Score over time: 0.673
# It means 41% success rate

print ("Final Q-Table Values")
print (np.round(Q,3))
# Final Q-Table Values
# [[ 0.771  0.015  0.015  0.015]
#  [ 0.002  0.002  0.     0.589]
#  [ 0.     0.339  0.003  0.002]
#  [ 0.002  0.     0.002  0.411]
#  [ 0.828  0.005  0.002  0.   ]
#  [ 0.     0.     0.     0.   ]
#  [ 0.     0.     0.07   0.   ]
#  [ 0.     0.     0.     0.   ]
#  [ 0.006  0.006  0.     0.916]
#  [ 0.006  0.945  0.     0.   ]
#  [ 0.982  0.     0.003  0.001]
#  [ 0.     0.     0.     0.   ]
#  [ 0.     0.     0.     0.   ]
#  [ 0.     0.     0.963  0.   ]
#  [ 0.     0.995  0.     0.   ]
#  [ 0.     0.     0.     0.   ]]
