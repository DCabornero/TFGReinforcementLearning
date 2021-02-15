import numpy as np
import pandas as pd

import random
from scipy.special import softmax

from Bandit import Bandit


# Algoritmo exp3.
class Exp3(Bandit):
    # alpha: constante de exploración
    def __init__(self,ratings,alpha=0.1):
        super().__init__(ratings)
        self.alpha = alpha

    def add_itemArms(self):
        super().add_itemArms()
        # Valor de pesos para cada brazo
        self.arms['Weight'] = np.ones((len(self.arms.index)))

    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        weights = self.arms.loc[availableArms,['Weight']].to_numpy()[:,0]
        totalWeights = np.sum(weights)

        probs = (1-self.alpha)*weights/totalWeights + self.alpha/len(availableArms)
        chosen = np.random.choice(np.arange(len(availableArms)),p=probs)
        self.probChosen = probs[chosen]
        return availableArms[chosen]

    def update_preference(self,item,reward):
        peso = self.arms.loc[item,'Weight']
        exponente = (self.alpha/len(self.listItems))*(reward/self.probChosen)
        self.arms.loc[item,'Weight'] = peso*np.exp(exponente)

    def item_hit(self,item):
        super().item_hit(item)
        self.update_preference(item,1)

    def item_fail(self,item):
        super().item_fail(item)
        self.update_preference(item,0)

    def item_miss(self,item):
        super().item_miss(item)
        self.update_preference(item,0)
