# -*- coding: utf-8 -*-
from marioGA import *

def play():
    if len(sys.argv) > 1:
        fname = sys.argv[1]

        textTerminal.append("Playing "+fname)
        bill_murray = NeuralNetwork(state_size, action_size)
        bill_murray.load(fname)
        fit_bill, x_bill = fitness(bill_murray)
        textTerminal.append("x: {}, score: {}".format(x_bill, fit_bill))
        printInfos(textTerminal)

if __name__ == "__main__":
    play()