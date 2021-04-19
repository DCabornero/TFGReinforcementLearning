import numpy as np
import pandas as pd

import random

from abstract.Bandit import Bandit

# Bandido que escoge un brazo al azar
class RandomBandit(Bandit):
    # Sabiendo los items que ya ha visto el usuario, se devuleve otro aleatoriamente
    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        return np.random.choice(availableArms)

    def __str__(self):
        return 'Random'
