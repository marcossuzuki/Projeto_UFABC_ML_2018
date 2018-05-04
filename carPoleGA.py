"""
Serie de funcoes que implementa um algoritmo genetico para solucionar um problema de controle do jogo CartPole 
"""
import gym
import random
import numpy as np
import pandas as pd
from functools import reduce
from operator import add
from statistics import median, mean
from collections import Counter
from keras.models import Sequential # cria uma pilha de camadas de redes neurais
from keras.layers import Dense, Activation
from keras import initializers
from keras import optimizers

env = gym.make("CartPole-v1")
N_FRAMES = 500

def individuo():
    model = Sequential()
    model.add(Dense(units=4, input_dim=4))
    model.add(Activation("relu"))
    model.add(Dense(units=4))
    model.add(Activation("relu"))
    model.add(Dense(units=2))
    model.add(Activation("relu"))


    return model

def initial_population(size):
    ' Define uma populacao inicial de redes neurais, dado o tamanho de entrada size'
    pop = []
    for i in range(size):
        model = individuo()
        fit = fitness(model)
        
        pop.append((fit,model))
    return pop

def crossover(pop, mother, father):
    "Aplica crossover de um ponto dado a populacao e dois parametros desta, denominados mother e father (inteiros)"
    population = [a[1] for a in pop]
    for _ in range(3):
        point = random.sample([0,1,3,2,4,5], 1)
        point = point[0]
        #print(point)
        w_mother = population[mother].get_weights()
        w_father = population[father].get_weights()

        if point==0 or point==2 or point==4:
            inside_point = random.sample([0,1,2,3], 1)
            inside_point = inside_point[0]

            w_new_mother = w_mother[point][inside_point]
            w_new_father = w_father[point][inside_point]

            w_mother[point][inside_point] = w_new_father
            w_father[point][inside_point] = w_new_mother
        else:
            w_new_mother = w_mother[point]
            w_new_father = w_father[point]
            w_mother[point] = w_new_father
            w_father[point] = w_new_mother
    
    return np.asarray([w_mother, w_father])

def mutate(weights):
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

def fitness(model):
    " Retorna o fitness para cada modelo dado como entrada. Fitness eh definido atraves do score obtido pela rede"
    env.reset()
    score = 0
    prev_obs = []
    scores = []
    for _ in range(10):
        env.reset()
        score = 0
        for _ in range(N_FRAMES):
            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(pd.DataFrame(prev_obs).T))

            new_observation, reward, done, info = env.step(action)
            
            score += reward
            prev_obs = new_observation
            if done: break
        scores.append(score)
    summed = sum(scores)
    #print(summed)
    return summed /float(len(scores))

def grade(population):
    " Fitness medio da populacao "
    summed = reduce(add, (fitness(observation) for observation in population))
    return summed / float(len(population))

def evolve(population, retain_lenght=0.4, random_select=0.1, max_pop=10):
    " Evolui a populacao e retorna uma nova geracao"
    new_weights = []   

    graded = population
    # Aplica o sort baseado nos scores
    graded = sorted(population, key=lambda x: x[0], reverse=True)

    # Aplicado selecao direta, mantemos uma taxa baseado na variavel retain_lenght da populacao de entrada
    retained = int(len(graded)*retain_lenght)
    parents = graded[:retained]
    print(parents)

    parents_length = len(parents)
    desired_length = max_pop - parents_length
    children = [] 

    # Mantem alguns dos individuos de forma aleatoria
    if random_select > random.random():
        individual = individuo()
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
            male = individuo()
            female = individuo()
                
                
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

def teste(model):
    scores = []
    choices = []
    score_requirement = 50
    env = gym.make("CartPole-v1")   
    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(N_FRAMES):
            env.render()
            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(pd.DataFrame(prev_obs).T))

            new_observation, reward, done, info = env.step(action)
            
            score += reward
            choices.append(action)
            prev_obs = new_observation
            if done: break
        scores.append(score)
        print("Jogo finalizado com score: {}!".format(score))
    env.close()
    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    print(score_requirement)