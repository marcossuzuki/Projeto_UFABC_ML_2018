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
  if (dx>0 and action == 130):
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
    def __init__(self, state_size, action_size, no_ones_initializer=False):
        self.state_size = state_size
        self.action_size = action_size
        self.no_ones_initializer = no_ones_initializer
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
        #weights[0][:,-1]=np.array([random.uniform(-.5,.5)])
        weights[1] = range(9,4825)
        #for i in range(9,4825):
        #    weights[1][i-9] = i
        #weights[1][4816] = 0
        model.set_weights(weights)
        model.add(Dense(self.action_size, activation='relu'))
        if self.no_ones_initializer:
            weights = model.layers[1].get_weights()
            weights[0] = 1*np.ones((4816,self.action_size))
            weights = model.layers[1].set_weights(weights)
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
    
def population(size_population, state_size, action_size):
    ' Define uma populacao inicial de redes neurais, dado o tamanho de entrada size'
    pop = []
    for i in range(size_population):
        agent = DQNAgent(state_size, action_size)
        fit, x = fitness(agent)
        
        pop.append((fit,agent,x))
    return pop

def inciarEmulador():
    global rle
    if rle != 0:
        rle.loadROM('super_mario_world.smc', 'snes')
    else:
        rle = loadInterface(True)
    
    getState(rle.getRAM(), radius)
    _=os.system("clear")

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

def crossover(pop, mother, father):
    "Aplica crossover de um ponto dado a populacao e dois parametros desta, denominados mother e father (inteiros)"
    population = [a[1] for a in pop]
    for _ in range(500):
        point = 2#random.sample([0,2], 1)[0]

        #print(point)
        w_mother = population[mother].get_weights()
        w_father = population[father].get_weights()

        '''
        if point == 0:
            for _ in range(150):
                inside_point = random.choice(range(170))

                w_new_mother = w_mother[point][inside_point]
                w_new_father = w_father[point][inside_point]

                w_mother[point][inside_point] = w_new_father
                w_father[point][inside_point] = w_new_mother
        '''
        if point==2:# or point==4:
            inside_point = random.sample(range(4816), 1)[0]

            w_new_mother = w_mother[point][inside_point]
            w_new_father = w_father[point][inside_point]

            w_mother[point][inside_point] = w_new_father
            w_father[point][inside_point] = w_new_mother
        
    return np.asarray([w_mother, w_father])

def evolve(population, retain_lenght=0.4, random_select=0.1, max_pop=50, number_gen=1):
    " Evolui a populacao e retorna uma nova geracao"

    new_weights = []

    graded = population
    # Aplica o sort baseado nos scores
    graded = sorted(population, key=lambda x: x[0], reverse=True)

    # Aplicado selecao direta, mantemos uma taxa baseado na variavel retain_lenght da populacao de entrada
    retained = int(len(graded)*retain_lenght)
    parents = graded[:retained]

    pop_gen = [ind[1] for ind in parents]
    for i, element in enumerate(pop_gen):
        element.save("marioGA.ind"+str(i))
    
    for i in (range(len(parents))):
        print(parents[i])

    parents_length = len(parents)
    desired_length = max_pop - parents_length
    children = [] 

    # Mantem alguns dos individuos de forma aleatoria
    if random_select > random.random():
        individual = DQNAgent(state_size, action_size)
        fit_individual = fitness(individual)
        parents.append((fit_individual, individual))

    #if random.uniform(0, 1) > 0.85:
     #   mutated = random.randint(0, parents_length-1)
      #  mutate(parents[mutated])

    #print(parents)

    # Calcula a quantidade de crossover necessaria para manter a taxa com max_pop
    
    # Add children, which are bred from two remaining networks.
    while len(children) < desired_length:
        # Get a random mom and dad.
        male = random.randint(0, parents_length-1)
        female = random.randint(0, parents_length-1)
        
            
        # Assuming they aren't the same network...
        if male != female:
            new_weights1 = crossover(parents, female, male)
            x_male = parents[male][2]
            x_female = parents[female][2]
            male = DQNAgent(state_size, action_size)
            female = DQNAgent(state_size, action_size)
   
            female.set_weights(mutate(new_weights1[0], x_female))
            male.set_weights(mutate(new_weights1[1], x_male))
                
            if len(children) < desired_length:
                fit_male, x_male = fitness(male)
                fit_female, x_female = fitness(female)
                if fit_male > fit_female:
                    children.append((fit_male, male, x_male))
                else:
                    children.append((fit_female, female, x_female))
                    
    parents.extend(children)

    return parents

def mutate(weights, x):
    for xi in range(x-80,len(weights[2])):
        try:
            for yi in range(len(weights[2][xi])):
                if random.uniform(0, 1) > 0.40:
                    change = random.uniform(-0.50,0.50)
                    weights[2][xi][yi] += change
        except:
            if random.uniform(0, 1) > 0.40:
                change = random.uniform(-0.50,0.50)
                weights[2][xi] += change
    return weights

def generations(pop, n_gen=10):
    initial_population = pop
    for i in range(n_gen):
        print("Geração "+str(i))
        new_pop = evolve(initial_population, random_select=0.01, number_gen = i)
        initial_population = new_pop
    return new_pop

def main():
    initial_population = population(50, state_size, action_size)
    evolved = generations(initial_population, 1000)

    return evolved

if __name__ == "__main__":
    print(main())
