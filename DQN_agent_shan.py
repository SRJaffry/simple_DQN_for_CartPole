# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 04:04:03 2020
@author: Shan Jaffry
"""
import numpy as np
import gym
import random
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from collections import deque

import matplotlib.pyplot as plt
#%matplotlib inline 

env = gym.make('CartPole-v0')

EPISODES = 200
BATCH_SIZE = 64
DISCOUNT = 0.95
UPDATE_TARGET_EVERY = 5
STATE_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n
SHOW_EVERY = 50

class DQNAgents:
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_memory = deque(maxlen = 2000)
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self._build_model()
        self.target_model = self.model
        
        self.target_update_counter = 0
        print('Initialize the agent')
        
    def _build_model(self):
        model = Sequential()
        model.add(Dense(20, input_dim = self.state_size, activation = 'relu'))
        model.add(Dense(10, activation = 'relu'))
        model.add(Dense(self.action_size, activation = 'linear'))
        model.compile(loss = 'mse', optimizer = Adam(lr = 0.001))
        
        return model

    def update_replay_memory(self, current_state, action, reward, next_state, done):
        self.replay_memory.append((current_state, action, reward, next_state, done))
        
    def train(self, terminal_state):
        
        # Sample from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        
        #Picks the current states from the randomly selected minibatch
        current_states = np.array([t[0] for t in minibatch])
        current_qs_list= self.model.predict(current_states) #gives the Q value for the policy network
        new_state = np.array([t[3] for t in minibatch])
        future_qs_list = self.target_model.predict(new_state)
        
        X = []
        Y = []
        
        # This loop will run 32 times (actually minibatch times)
        for index, (current_state, action, reward, next_state, done) in enumerate(minibatch):
            
            if not done:
#                new_q = reward + DISCOUNT * np.max(future_qs_list)
                new_q = reward + DISCOUNT * np.max(future_qs_list[index])
            else:
                new_q = reward
                
            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            
            X.append(current_state)
            Y.append(current_qs)
        
        # Fitting the weights, i.e. reducing the loss using gradient descent
        self.model.fit(np.array(X), np.array(Y), batch_size = BATCH_SIZE, verbose = 0, shuffle = False)
        
       # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1
            
        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    
    def get_qs(self, state):
        # https://stackoverflow.com/questions/55436040/keras-mismatched-array-shape-with-model-predict?rq=1
        # Keras expects the data to be implicitly (N, D) where N is the batch size and D is the number of features.
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
            

'''
Main function starts here
'''

agent = DQNAgents(STATE_SIZE, ACTION_SIZE)

Episodic_reward = []

for e in range(EPISODES):
    
    done = False
    current_state = env.reset()
    time = 0 
    total_reward = 0
    while not done:
        if np.random.random() > agent.epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = env.action_space.sample()
        
        next_state, reward, done, _ = env.step(action)

        agent.update_replay_memory(current_state, action, reward, next_state, done)
        
        if len(agent.replay_memory) < BATCH_SIZE:
            pass
        else:
            agent.train(done)
            
        time+=1    
        current_state = next_state
        total_reward += reward
        
    print(f'episode : {e}, steps {time}, epsilon : {agent.epsilon}')
    Episodic_reward.append(time)
    
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    

plt.plot(Episodic_reward)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.legend(['Reward'])
plt.show()
