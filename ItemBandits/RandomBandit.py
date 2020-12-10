import numpy as np
import pandas as pd

import random

from Bandit import Bandit

# Bandido que escoge un brazo al azar
class RandomBandit(Bandit):
    def __init__(self,ratings):
        super().__init__(ratings)

    # Sabiendo los items que ya ha visto el usuario, se devuleve otro aleatoriamente
    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        return np.random.choice(availableArms)
