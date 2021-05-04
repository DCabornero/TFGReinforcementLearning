import numpy as np
import pandas as pd

import random
from scipy.special import softmax

from abstract.Bandit import Bandit


# Algoritmo exp3.
class Exp3(Bandit):
    name = 'EXP3'
    # alpha: constante de exploración
    def __init__(self):
        super().__init__()
        self.alpha = 1

    def add_itemArms(self):
        super().add_itemArms()
        # Valor de pesos para cada brazo
        self.arms['Weight'] = np.ones((len(self.arms.index)))
        # Valor de recompensa esperada para cada brazo
        # self.arms['Reward'] = np.zeros((len(self.arms.index)))
        # self.oldEpsilon = 1/len(self.arms.index)
        self.epochs = 0

    def __str__(self):
        return self.name

    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        weights = self.arms.loc[availableArms,['Weight']].to_numpy()[:,0]
        totalWeights = np.sum(weights)

        probs = (1-self.alpha)*weights/totalWeights + self.alpha/len(availableArms)
        chosen = np.random.choice(np.arange(len(availableArms)),p=probs)
        self.probs = probs
        self.probChosen = probs[chosen]
        return availableArms[chosen]

    # def select_arm(self,viewed):
    #     availableArms = self.available_arms(viewed)
    #     numArms = len(availableArms)
    #     self.epochs += 1
    #     # Nuevo epsilon
    #     epsilon = np.min([1/numArms,np.sqrt(np.log(numArms)/numArms*self.epochs)])
    #     # Cálculo de probabilidades
    #     rewards = self.arms.loc[availableArms,'Reward'].to_numpy()
    #     values = np.exp(self.oldEpsilon*rewards)
    #     probs = (1-numArms*epsilon)*values/np.sum(values) + epsilon
    #     # Acutalización de oldEpsilon
    #     self.oldEpsilon = epsilon
    #     # Elección de brazo
    #     ind = np.random.choice([i for i in range(len(availableArms))],p=probs)
    #     self.prob = probs[ind]
    #     return availableArms[ind]


    def update_preference(self,item,reward):
        peso = self.arms.loc[item,'Weight']
        exponente = (self.alpha/len(self.probs))*(reward/self.probChosen)
        self.arms.loc[item,'Weight'] = peso*np.exp(exponente)
        self.epochs += 1
        self.alpha = np.exp(-self.epochs/(10*len(self.listItems)))

        # self.arms.at[item,'Reward'] += reward/self.prob

    def item_hit(self,item):
        super().item_hit(item)
        self.update_preference(item,1)

    def item_fail(self,item):
        super().item_fail(item)
        self.update_preference(item,0)

    def item_miss(self,item):
        super().item_miss(item)
        self.update_preference(item,0)
