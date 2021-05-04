import numpy as np
import pandas as pd

from abstract.Bandit import Bandit

class CNAME(Bandit):
    name = 'CNAME'
    def __init__(self,w=1):
        super().__init__()
        self.w = w

    def __str__(self):
        return self.name

    def add_itemArms(self):
        super().add_itemArms()
        self.arms['Q'] = np.zeros((len(self.arms.index)))
        self.arms['N'] = np.zeros((len(self.arms.index)))
        self.epochs = 0

    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)

        # Cálculo del parámetro p
        qlist = self.arms.loc[availableArms,'Q'].to_numpy()
        arg = np.argmin(qlist)
        m = self.arms.loc[:,'N'].to_numpy()[arg]
        p = self.w/(self.w+m*m)

        # Elección de exploración/explotación
        if np.random.random_sample() > p:
            return availableArms[np.argmax(qlist)]
        else:
            return np.random.choice(availableArms)

    # Estaría bien actualizar solo las que entran en el bombo
    def update_preference(self,item,reward):
        self.arms.at[item,'N'] += 1
        self.arms.at[item,'Q'] += 1/self.arms.at[item,'N'] * (reward - self.arms.at[item,'Q'])

    def item_hit(self,item):
        super().item_hit(item)
        self.update_preference(item,1)

    def item_fail(self,item):
        super().item_fail(item)
        self.update_preference(item,0)

    def item_miss(self,item):
        super().item_miss(item)
        self.update_preference(item,0)
