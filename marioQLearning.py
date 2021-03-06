#!/usr/bin/env python
# marioQLearning.py
# Author: Fabrício Olivetti de França
#
# A simple Q-Learning Agent for Super Mario World
# using RLE

import sys
from rle_python_interface.rle_python_interface import RLEInterface
import numpy as np
from numpy.random import uniform, choice, random

import time
from rominfo import *
from utils import *

# parâmetros do Q-Learning
l_rate = 0.8
gamma  = 0.6
thr    = 0.9
radius = 10
  
def getReward(reward, gameOver, action, dx):
  R = 0 
  
  # +0.5 for stomping enemies or getting itens/coins
  if reward > 0 and not gameOver:
    R += 0.3*np.log(reward)
 
  # incentiva andar pra direita
  if action == 128 and dx > 0:
    R += 0.5
  # e pular correndo
  elif action == 130 and dx > 0:
    R += 0.5

  if gameOver:
    R  -= 2.0
  else:
    R  += 0.1 
    
  return R
   
# Q-Learning      
def train():
  rle = loadInterface(False)
  
  Q, ep, maxActions = getStoredQ()

  total_reward, total_my_reward = 0, 0
  state, x, y = getState(rle.getRAM(), radius)
  randomOrnew = random()
  it = 0
  history = []
    
  for episode in range(10): 
      rle.saveState()
      while not rle.game_over():
      
        # escolha uma acao
        # aleatoria com thr % de chances
        if random() > thr:
          if randomOrnew > 0.5:
            a = actions_list[choice(len(actions_list))]
          else:
            # pega uma ação nunca escolhida para diversificar
            a = actions_list[getNewActionDet(Q, state)]
        # deterministica (1-thr)% de chances ou a cada dez episodios
        else:
          a = actions_list[getBestActionDet(Q, state)]

        # aplica a acao e monta a chave da tabela e pega o proximo estado
        reward = performAction(a, rle)
          
        current_key = f'{state},{a}'
        history.append(current_key)
           
        state, xn, yn = getState(rle.getRAM(), radius)
            
        # contabiliza os ganhos
        R = getReward(reward, rle.game_over(), a, xn-x)
        x = xn
        total_reward += reward
        total_my_reward += R
        
        # verifica a melhor previsao de valor
        best  = getBestActionDet(Q, state)
        key   = f'{state},{best}'
        #R2 = getReward(reward, rle.game_over(), best, xn-x)
        futureQ, t = Q.get(key, (0.0, 0))
        if rle.game_over():
          futureQ = 0.0
          R = 0.0

        # atualiza o valor de Q para esse estado
        Qval, t = Q.get(current_key, (0.0, 0))
        alpha = (l_rate/(t+1))
        newQval = (1-alpha)*Qval + alpha*(R + gamma*futureQ)
        Q[current_key] = (newQval, t+1)
        
        it += 1
        if it > maxActions:
          maxActions = it
        
      for h in history[:-4]:
        Qval, t = Q[h]
        alpha = (l_rate/(t+1))
        newQval = (1-alpha)*Qval + alpha*0.1*np.log2(xn)
        Q[h] = (newQval, t)
        
      for h in history[-4:]:
        Qval, t = Q[h]
        Q[h] = (Qval-0.1, t)  
      
      rle.loadState()
      
  ep += 1
  fw = open('Q.pkl', 'wb')
  pickle.dump((Q, ep, maxActions), fw)
  print(f'{ep} ({thr}, {it}): REWARD: {total_reward}, {total_my_reward} {xn}!!!!!!!!!!!!!!!!!!')
  fw.close()
  print(Q)
       
  
  return total_reward


def main():
  r = train()
  time.sleep(2)
    
if __name__ == "__main__":
  main()

  
