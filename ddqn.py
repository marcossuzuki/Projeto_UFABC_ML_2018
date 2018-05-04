# -*- coding: utf-8 -*-
import random
#import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

import sys
from rle_python_interface.rle_python_interface import RLEInterface
#from numpy.random import uniform, choice, random

import time
from rominfo import *
from utils import *
radius = 6

def getReward(reward, gameOver, action, dx):
  R = 0 
  
  # +0.5 for stomping enemies or getting itens/coins
  if reward > 0 and not gameOver:
    R += 0.3*np.log(reward)
 
  # incentiva andar pra direita
  if (dx>0):
    R += 0.5
  # e pular correndo
  if (action == 129 or action == 131 or action == 384 or action == 386):
    R += 0.1

  if gameOver:
    R  -= 5.0
    
  return R

EPISODES = 10000


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.950    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 1.00
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.dict_models = {}
        self.dict_target_models = {}
        self.update_target_model()
        self.update_model()

    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(10, input_dim=self.state_size, activation='linear'))
        model.add(Dense(2, activation='sigmoid'))
        #model.add(Dense(2, activation='sigmoid'))
        model.add(Dense(self.action_size, activation='sigmoid'))
        model.compile(loss=self._huber_loss, #sample_weight_mode='temporal',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        
    def update_model(self, count=0):
        if count in self.dict_models:
            self.model = self.dict_models[count]
            self.target_model = self.dict_target_models[count]

    def remember(self, state, action, R, reward, next_state, gameOver, x):
        self.memory.append((state, action, R, reward, next_state, gameOver, x))

    def act(self, state, x, threshold):
        if np.random.rand() <= self.epsilon and x > max((threshold-100),0):# and x < (threshold+50):
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size, gameOver, count):
        #minibatch = random.sample(self.memory, batch_size)
        if gameOver:
            minibatch = deque(self.memory, batch_size)
            for state, action, R, reward, next_state, _, x in minibatch:
                target = self.model.predict(state)
                self.memory.pop()
                target[0][action] = R
                self.model.fit(state, target, epochs=1, verbose=0)
            for state, action, R, reward, next_state, _, x in self.memory:
                target = self.model.predict(state)
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = R + self.gamma * t[np.argmax(a)]
                self.model.fit(state, target, epochs=1, verbose=0)
                #self.weights[x] = self.model.get_weights()
            
        else:
            for state, action, R, reward, next_state, _, x in self.memory:
                target = self.model.predict(state)
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = R + self.gamma * t[np.argmax(a)]
                self.model.fit(state, target, epochs=1, verbose=0)
                #self.weights[x] = self.model.get_weights()
            #if count not in self.dict_models:
            #    self.dict_models[count] = self.model
            #    self.dict_target_models[count] = self.target_model
            
        #if self.epsilon > self.epsilon_min:
        #    self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    rle = loadInterface(True)
    Q, ep, maxActions = getStoredQ()

    total_reward, total_my_reward = 0, 0
    state, x, y = getState(rle.getRAM(), radius)
    randomOrnew = np.random.random()
  
    state_size = (radius*2+1)*(radius*2+1)+2
    action_size = len(actions_list)
    
    agent = DQNAgent(state_size, action_size)
    
    #agent.load("./save/ddqn.h5")
    batch_size = 7
    salvo = False
    x_velho=0
    threshold = 0
    for e in range(EPISODES):
        rle.saveState()
        state, x, y = getState(rle.getRAM(), radius)
        state += ','+str(x)+','+str(y)
        state=state.split(',')
        state = np.reshape(np.array(state), [1, state_size])
        total_my_reward = 0 
        parado = False
        count = 0
        while not rle.game_over():
            
            state, x, y = getState(rle.getRAM(), radius)
            state += ','+str(x)+','+str(y)
            state=state.split(',')
            state = np.reshape(np.array(state), [1, state_size])
            
            a = agent.act(state, x, threshold)
            action = actions_list[a]
            
            reward = performAction(action, rle)
            
            next_state, xn, yn = getState(rle.getRAM(), radius)
            
            # contabiliza os ganhos
            R = getReward(reward, rle.game_over(), action, xn-x)
            '''if (xn-x)<1:
                count+=1
                if count>20:
                    parado = True
            '''
            x = xn
            
            #if x>170 and not salvo:
               #rle.saveState()
               #salvo = True
            total_reward += reward
            total_my_reward += R #+ reward
            
            #if rle.game_over():
            #    R = -1.0*100.0
            #reward = a if not rle.game_over() else -10
            next_state += ','+str(xn)+','+str(yn)
            
            next_state = next_state.split(',')
            next_state = np.reshape(np.array(next_state), [1, state_size])

            agent.remember(state, a, R, total_my_reward, next_state, rle.game_over(), x)

            #state = next_state
            #print(state)
            '''
            if x_velho < x or reward>0:
                agent.update_target_model()
                x_velho = x
            '''
            
            
            #print("episode: {}/{}, score: {}, e: {:.2}, x{}"
            #          .format(e, EPISODES, total_my_reward, agent.epsilon, x))
            '''if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break'''
            #agent.update_target_model()
            if len(agent.memory) > batch_size and reward>0:
                agent.replay(batch_size, rle.game_over(),count)
                agent.update_target_model()
                #agent.memory.clear()
            elif rle.game_over():
                agent.replay(batch_size, rle.game_over(), count)
                agent.update_target_model()
                #agent.memory.clear()
                threshold = x
            '''
            if len(agent.memory)%batch_size==0 :
                agent.replay(batch_size, rle.game_over(),count)
                count+=1
                agent.update_model(count)
                print(count)
            '''    
            #if len(agent.memory) > batch_size:
            #    agent.replay(batch_size)
            
        agent.memory.clear()
        #count=0
        #agent.update_model(count)
        #agent.update_weight()
        #agent.replay(batch_size)
        agent.update_target_model()
        agent.save("./save/ddqn.h5")
        
        print("episode: {}/{}, score: {}, e: {:.2}, x = {}"
                  .format(e, EPISODES, total_my_reward, agent.epsilon, x))
        rle.loadState()
        #rle = loadInterface(True)
