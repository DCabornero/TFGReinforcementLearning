import numpy as np
import pandas as pd

import random

from EpsilonGreedy import EpsilonGreedy

class UCB(EpsilonGreedy):
    def __init__(self,alpha=None,c=2):
        super().__init__(alpha=alpha)
        self.c = c

    def add_itemArms(self):
        super().add_itemArms()
        self.epochs = 0

    def new_reward(self,item,reward):
        super().new_reward(item,reward)
        self.epochs += 1

    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)

        accuracy = self.arms.loc[availableArms,['Accuracy']].to_numpy()[:,0]
        # Número de épocas que se han dedicado a cada item no explorado
        itemsEpochs = self.arms.loc[availableArms,['Epochs']].to_numpy()[:,0]

        zeros = availableArms[itemsEpochs == 0]
        # Si algún brazo no se ha elegido aún, se da máxima prioridad
        if len(zeros) > 0:
            return random.choice(zeros)
        else:
            preds = accuracy + self.c*np.sqrt(np.divide(np.log(self.epochs),itemsEpochs))
            index = np.argmax(preds)
            return availableArms[index]
