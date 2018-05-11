# -*- coding: utf-8 -*-
import random
#import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
import pickle

import sys
from rle_python_interface.rle_python_interface import RLEInterface
#from numpy.random import uniform, choice, random

import time
from rominfo import *
from utils import *
radius = 6

rle = 0
state_size = 1 #(radius*2+1)*(radius*2+1)+2
action_size = len(actions_list)

def getReward(reward, gameOver, action, dx):
  R = 0 
  
  # +0.5 for stomping enemies or getting itens/coins
  if reward > 0 and not gameOver:
    R += 3.0*np.log(reward)
 
  # incentiva andar pra direita
  if (dx>0):
    R += 0.5
  # e pular correndo
  if (action == 129 or action == 131 or action == 384 or action == 386):
    R += 0.1

  if gameOver:
    R  -= 5.0
    
  return R

def custom_activation(x):
    return K.relu(K.sigmoid(x+1)-K.sigmoid(x-1)-0.4)*15.38
    #return K.relu(1-K.relu(x))

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
    
    def get_weights(self):
        return self.model.get_weights()
    def set_weights(self, weights):
        self.model.set_weights(weights)


    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(4816, input_dim=self.state_size, activation=custom_activation))
        weights = model.get_weights()
        weights[0] = -1*np.ones((1,4816))
        weights[1] = range(9,4825)
        model.set_weights(weights)
        
        #model.add(Dense(4, activation='relu'))
        #model.add(Dense(2, activation='sigmoid'))
        model.add(Dense(self.action_size, activation='relu'))
        weights = model.layers[1].get_weights()
        weights[0] = 1*np.ones((4816,self.action_size))
        model.layers[1].set_weights(weights)
        model.compile(loss=self._huber_loss, #sample_weight_mode='temporal',
                      optimizer=Adam(lr=self.learning_rate))
        return model


    def act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

def inciarEmulador():
    global rle
    rle = loadInterface(True)
    getState(rle.getRAM(), radius)
       
def fitness(agent_):
    inciarEmulador()

    total_reward, total_my_reward = 0, 0
    state, x, y = getState(rle.getRAM(), radius)
    
    state = np.reshape(np.array(x), [1, state_size])
    total_my_reward = 0
    dx=0
    c = 0
    while not rle.game_over() and c<50:
        #state, x, y = getState(rle.getRAM(), radius)
        state = np.reshape(np.array(x), [1, state_size])
                
        a = agent_.act(state)
        action = actions_list[a]
                
        reward = performAction(action, rle)
                
        next_state, xn, yn = getState(rle.getRAM(), radius)
                
        # contabiliza os ganhos
        R = getReward(reward, rle.game_over(), action, xn-x)
        dx=xn-x
        if dx == 0:
            c+=1

        x = xn

        total_reward += reward
        total_my_reward += R #+ reward
                
    print("score: {}, x = {}".format(total_my_reward, x))
    return total_my_reward, x

def generations(pop, n_gen=10):
    initial_population = pop
    for i in range(n_gen):
        
        print("\nDia da marmota "+str(i+1))
        
        fitness(pop[0][1])
 

def mutate(weights, x):
    for xi in range(x-15,x+15):
        try:
            for yi in range(len(weights[2][xi])):
                #change = random.uniform(-1,1)
                #weights[2][xi][yi] += change
                if yi == 0 :
                    weights[2][xi][yi] = 0
                else:
                    weights[2][xi][yi] = 1
                
        except:
            if random.uniform(0, 1) > 0.50:
                change = random.uniform(-1,1)
                weights[2][xi] += change
    return weights

def main():
    if len(sys.argv) > 1:
        fname = sys.argv[1]

    print("-------------------------------------------------")
    print("Dia da marmota 0")
    print("-------------------------------------------------")
    bill_murray = DQNAgent(state_size, action_size)
    bill_murray.load(fname)
    fit_bill, x_bill = fitness(bill_murray)
    bill_quality = []
    bill_quality.append((fit_bill, bill_murray, x_bill))


    evolved = generations(bill_quality, 10000)

    

if __name__ == "__main__":
    main()
