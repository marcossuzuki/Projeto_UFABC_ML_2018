# -*- coding: utf-8 -*-
import random
#import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
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

    #def gaussian(x):


    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(4816, input_dim=self.state_size, activation='relu'))
        weights = model.get_weights()
        weights[0] = np.ones((1,4816))
        weights[1] = range(-4825,-9)
        model.set_weights(weights)
        #model.add(Dense(4, activation='relu'))
        #model.add(Dense(2, activation='sigmoid'))
        model.add(Dense(self.action_size, activation='relu'))
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
        
        pop.append((fit,x,agent))
    return pop

def inciarEmulador():
    global rle
    if rle == 0:
        rle = loadInterface(False)
        getState(rle.getRAM(), radius)
        rle.saveState()
    else:
        rle.loadState()
        rle.saveState()

def fitness(agent_):
    inciarEmulador()

    total_reward, total_my_reward = 0, 0
    state, x, y = getState(rle.getRAM(), radius)
    
    state, x, y = getState(rle.getRAM(), radius)
    state = np.reshape(np.array(x), [1, state_size])
    total_my_reward = 0

    while not rle.game_over():
        state, x, y = getState(rle.getRAM(), radius)
        state = np.reshape(np.array(x), [1, state_size])
                
        a = agent_.act(state)
        action = actions_list[a]
                
        reward = performAction(action, rle)
                
        next_state, xn, yn = getState(rle.getRAM(), radius)
                
        # contabiliza os ganhos
        R = getReward(reward, rle.game_over(), action, xn-x)
        x = xn

        total_reward += reward
        total_my_reward += R #+ reward
                
    print("score: {}, x = {}".format(total_my_reward, x))
    return total_my_reward, x

def crossover(pop, mother, father):
    "Aplica crossover de um ponto dado a populacao e dois parametros desta, denominados mother e father (inteiros)"
    population = [a[1] for a in pop]
    for _ in range(3):
        point = random.sample([0,2], 1)[0]

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
            inside_point = random.sample([0,1,2,3], 1)[0]

            w_new_mother = w_mother[point][inside_point]
            w_new_father = w_father[point][inside_point]

            w_mother[point][inside_point] = w_new_father
            w_father[point][inside_point] = w_new_mother
    
    return np.asarray([w_mother, w_father])

def evolve(population, retain_lenght=0.4, random_select=0.1, max_pop=10, number_gen=1):
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
        element.save("./save2/marioGA_1.gen_"+str(number_gen)+"ind"+str(i))

    print(parents)

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
            male = DQNAgent(state_size, action_size)
            female = DQNAgent(state_size, action_size)
                
                
            female.set_weights(mutate(new_weights1[0]))
            male.set_weights(mutate(new_weights1[1]))
                
            if len(children) < desired_length:
                fit_male = fitness(male)
                fit_female = fitness(female)
                if fit_male > fit_female:
                    children.append((fit_male,male))
                else:
                    children.append((fit_female, female))
                    
    parents.extend(children)

    return parents

def mutate(weights,x):
    for xi in range(len(weights)):
        try:
            for yi in range(len(weights[xi])):
                if random.uniform(0, 1) > 0.50:
                    change = random.uniform(-0.50,0.50)
                    weights[xi][yi] += change
        except:
            if random.uniform(0, 1) > 0.50:
                change = random.uniform(-0.50,0.50)
                weights[xi] += change
    return weights

def generations(pop, n_gen=10):
    initial_population = pop
    for i in range(n_gen):
        new_pop = evolve(initial_population, number_gen = i)
        initial_population = new_pop
    return new_pop

def main():
    initial_population = population(5, state_size, action_size)
    '''
    for i in range(4):
        agent = initial_population[i][1]
        agent.load("./save/marioGA_1.gen_49ind" +str(0))
        fit = fitness(agent)
        initial_population[i]= (fit, agent)'''
    '''
    print("-------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------")
    print(initial_population)
   
    print("-------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------")
    '''
    evolved = generations(initial_population, 100)
#[(189.5894952099644, <__main__.DQNAgent object at 0x7ffb881eeb38>), (175.0894952099644, <__main__.DQNAgent object at 0x7ffbe7a2ddd8>), (167.5894952099644, <__main__.DQNAgent object at 0x7ffb98244c18>), (167.0894952099644, <__main__.DQNAgent object at 0x7ffb981160f0>)]

    #save_model(evolved)

    return evolved

def save_model(population):
    for individuo in population:
        with open(str(individuo[0])+".pickle", "wb") as fp:
            pickle.dump(individuo[1], fp)

if __name__ == "__main__":
    print(main())