import numpy as np
import pandas as pd

import random

from Bandit import Bandit

# Definición de funciones de distribución con soporte en [0,1] y media alpha/(alpha+beta)
# Recibe un array de alphas y betas y devuelve un valor aleatorio de cada distribución

# distribución beta
def Beta(alpha,beta):
    return np.random.beta(alpha,beta)

# distribución de Bernouilli
def Bernouilli(alpha,beta):
    rnd = np.random.rand(np.shape(alpha)[0])
    return np.array(rnd < alpha/(alpha+beta)).astype(int)

dict_func = {'Beta': Beta, 'Bernouilli': Bernouilli}
# Devuelve True si el soporte de una función es discreto.
disc_func = {'Beta':False, 'Bernouilli': True}
# Bandido que escoge un brazo al azar en función de una distribución
# dependiente del número de aciertos y errores de cada brazo
class ThompsonSampling(Bandit):
    def __init__(self,ratings,func='Beta',alpha=1,beta=1):
        super().__init__(ratings)
        self.alpha = alpha
        self.beta = beta
        self.func = dict_func.get(func)
        self.disc = disc_func.get(func)

    # Devuelve el índice del brazo seleccionado según el algoritmo
    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        results = self.arms.loc[availableArms,['Hits','Fails','Misses']].to_numpy()
        alpha = results[:,0] + self.alpha
        beta = np.sum(results[:,[1,2]],axis=1) + self.beta
        numbers = self.func(alpha,beta)
        # Si la función es discreta, se elige aleatoriamente entre los máximos valores
        # Si es continua, se puede utilizar "argmax" para hallar el máximo
        if self.disc:
            chosen = random.choice(np.argwhere(numbers == np.amax(numbers)))
        else:
            chosen = np.argmax(numbers)
        return availableArms[chosen]
