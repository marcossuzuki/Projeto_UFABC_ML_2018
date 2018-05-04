#!/usr/bin/env python3
# marioGA.py
# Author: Victor de Oliveira
#
# Série de funções que implementa algoritmos genéticos em redes neurais que aprendem a jogar o jogo Super Mario World

import random
import numpy as np
import pandas as pd
from keras.models import Sequential # cria uma pilha de camadas de redes neurais
from keras.layers import Dense, Activation
from keras import initializers
from keras import optimizers


def neural_model(input_size, output_size):

    model = Sequential()

    model.add(Dense(units=64, input_dim=input_size))
    model.add(Dropout(0.25))

    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.25))


    model.add(Dense(units=output_size, activation='softmax'))
    model.add(Dropout(0.25))

    return model

     model = Sequential()
    model.add(Dense(units=4, input_dim=4))
    model.add(Activation("relu"))
    model.add(Dense(units=4))
    model.add(Activation("relu"))
    model.add(Dense(units=2))
    model.add(Activation("relu"))
