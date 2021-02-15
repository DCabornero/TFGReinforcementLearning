import numpy as np
import pandas as pd

import random
from scipy.special import softmax

from Bandit import Bandit


# Algoritmo gradiente. Cada brazo tiene una probabilidad que es recalculada
# en cada época según los criterios habituales del descenso gradiente.
class GradientBandit(Bandit):
    # alpha: tasa de aprendizaje del descenso gradiente
    def __init__(self,ratings,alpha=0.1):
        super().__init__(ratings)
        self.alpha = alpha
        self.avgReward = 0
        self.epochs = 0

    def add_itemArms(self):
        super().add_itemArms()
        # Valor de preferencia para cada brazo
        self.arms['Preference'] = np.zeros((len(self.arms.index)))
        # Probabilidad de escoger un cierto brazo
        self.arms['Prob'] = np.ones((len(self.arms.index)))/len(self.arms.index)

    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        probs = self.arms.loc[availableArms,['Prob']].to_numpy()[:,0]
        probs /= np.sum(probs)

        chosen = np.random.choice(availableArms,p=probs)
        return chosen

    def update_preference(self,item,reward):
        auxVector = np.zeros((len(self.arms.index)))
        auxVector[np.where(self.arms.index == item)] = 1
        self.arms['Preference'] -= self.alpha*(reward-self.avgReward)*(self.arms['Prob']-auxVector)
        self.arms['Prob'] = softmax(self.arms['Preference'])
        self.epochs += 1
        self.avgReward += (reward-self.avgReward)/self.epochs

    def item_hit(self,item):
        super().item_hit(item)
        self.update_preference(item,1)

    def item_fail(self,item):
        super().item_fail(item)
        self.update_preference(item,0)

    def item_miss(self,item):
        super().item_miss(item)
        self.update_preference(item,0)
