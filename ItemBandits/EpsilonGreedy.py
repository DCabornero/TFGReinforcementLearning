import numpy as np
import pandas as pd

import random

from abstract.Bandit import Bandit


# Algoritmo epsilon-greedy. Se elige el algoritmo con mayor recompensa esperada
# con probabilidad 1-epsilon. Se escoge cualquier otro algoritmo con probabilidad
# epsilon.
class EpsilonGreedy(Bandit):
    name = 'Epsilon-Greedy'
    # epsilon: probabilidad con la que se explora
    # alpha: peso de la última recompensa dentro del promedio de recompensas (entre 0 y 1).
    # Si no se indica, se realiza el algoritmo epsilon-greedy clásico
    # initial: recompensa estimada de los brazos sin explorar (entre 0 y 1)
    def __init__(self,epsilon=0.1,alpha=None,initial=0):
        super().__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial = initial

    def __str__(self):
        return self.name

    def add_itemArms(self):
        super().add_itemArms()
        self.arms['Epochs'] = np.zeros((len(self.arms.index)))
        self.arms['Accuracy'] = np.ones((len(self.arms.index)))*self.initial

    # Calcula la recompensa estimada tras un paso. Recibe la recompensa estimada
    # actual, la nueva recompensa y el peso de dicha recompensa.
    def calcula_recompensa(self,acc,award,alpha):
        return acc + alpha*(award-acc)

    # Actualiza la recompensa estimada del brazo sabiendo la nueva recompensa.
    def new_reward(self,item,reward):
        acc = self.arms.loc[item,'Accuracy']
        if self.alpha:
            alpha = self.alpha
        else:
            suma = self.arms.loc[item,'Epochs']
            if suma == 0:
                alpha = 1
            else:
                alpha = 1/suma
        self.arms.loc[item,'Accuracy'] = self.calcula_recompensa(acc,reward,alpha)
        self.arms.loc[item,'Epochs'] += 1

    def item_hit(self,item):
        super().item_hit(item)
        self.new_reward(item,1)

    def item_fail(self,item):
        super().item_fail(item)
        self.new_reward(item,0)

    def item_miss(self,item):
        super().item_miss(item)
        self.new_reward(item,0)

    # Devuelve el índice del brazo seleccionado según el algoritmo
    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        accuracy = self.arms.loc[availableArms,'Accuracy'].to_numpy()

        # Hallamos el brazo de máxima precisión y lo separamos de choices
        bestIndex = np.argmax(accuracy)
        best = availableArms[bestIndex]
        availableArms = np.delete(availableArms,bestIndex)

        # Elegimos la mejor opción o una de las otras aleatoriamente según el criterio
        # explicado antes
        number = random.random()
        if(number < self.epsilon):
            return np.random.choice(availableArms)
        else:
            return best
