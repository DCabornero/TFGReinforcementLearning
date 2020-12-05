from abc import abstractmethod

import numpy as np
import pandas as pd

from Datos import Datos
from Arm import ArmItem

from sklearn.model_selection import train_test_split

import random

import matplotlib.pyplot as plt

from time import time

# Cada uno de los Arms tendrá un sistema de recomendación que permita
# dar un elemento a recomendar para un cierto usuario siguiendo un cierto algoritmo.
# El trainSet debe ser dado en formato de matriz numPy.
class Bandit:
    # ratings: CSV que contiene todos los ejemplos posibles con mínimo tres columnas:
    # - ID del usuario (default: userId)
    # - ID del item (default: itemId)
    # - valorción que ha dado dicho usuario a dicho item (default: rating)
    # movies: CSV que contiene la base de datos de los items con mínimo dos datos:
    # - ID del item (llamado itemId)
    # - nombre de los tags del item separados por barras (default: genres)
    def __init__(self,ratings):
        self.ratings = Datos(ratings)
        self.arms = None
        self.times = np.zeros((3))

    def add_itemArms(self):
        # Obtención de la lista de items
        itemsRep = self.ratings.extraeCols(['movieId'])[:,0]
        items = np.unique(itemsRep)

        funcs = list(map(lambda x: ArmItem(x), items))

        self.arms = pd.DataFrame({'Arm': funcs}, index=items)

    # Crea un diccionario que tiene a los usuarios por key y un array de sus
    # items ya valorados como valor
    def get_rated(self,trainSet):
        viewed = {}
        listUsers = np.unique(trainSet[:,0])
        for u in listUsers:
            itemsRated = trainSet[trainSet[:,0] == u,1]
            viewed[u] = itemsRated
        return viewed

    # Devuelve la lista de los items no valorados sabiendo cuales se han valorado ya
    def available_arms(self,viewed):
        keys = self.arms.index
        # availableKeys = np.setdiff1d(keys,viewed)

        arr = self.arms.to_numpy()[np.isin(keys,viewed,assume_unique=True,invert=True),0]

        return arr


    # Depende del algoritmo de selección concreto
    @abstractmethod
    def select_arm(self):
        pass

    # Queda run epoch
    def run_epoch(self,epochs=500,trainSize=0.1):
        index = 0
        epoch = 0
        totalhits = 0


        cols = ['userId','movieId','rating']
        train, test = train_test_split(self.ratings.extraeCols(cols), train_size=trainSize)

        viewed = self.get_rated(train)
        self.add_itemArms()

        plt.close()
        plt.xlabel('Épocas')
        plt.ylabel('Aciertos')
        results = [[],[]]

        listUsers = np.unique(train[:,0])
        numUsers = len(listUsers)
        random.shuffle(listUsers)

        # Comienzan a correr las épocas
        while epoch < epochs:
            target = listUsers[index]
            t0 = time()
            arm = self.select_arm(viewed)
            t1 = time()
            self.times[0] += t1-t0
            # Se recomiendo un item
            item = arm.rec_item()

            # Comprobamos si tenemos la recomendación del item en el testSet
            mask = np.logical_and(test[:,0] == target,test[:,1] == item)

            # Si hemos encontrado un resultado, lo valoramos como hit o fail
            if(np.count_nonzero(mask) > 0):
                # epoch += 1
                test, hit = test[np.logical_not(mask)], test[mask]
                # De momento, el umbral de valoración hit/fail es 3
                if hit[0,2] >= 3:
                    arm.hits += 1
                    totalhits += 1
                else:
                    arm.fails += 1
            else:
                arm.misses += 1
                viewed[target] = np.append(viewed[target],[item])

            results[0].append(epoch)
            results[1].append(totalhits)

            index += 1
            # Cuando agotamos los usuarios los barajamos y volvemos a empezar
            if index == numUsers:
                random.shuffle(listUsers)
                index = 0

            epoch += 1
        print(self.times)
        plt.plot(results[0],results[1])
        plt.savefig('hits.png')

# Algoritmo epsilon-greedy. Se elige el algoritmo con mayor tasa de hits
# con probabilidad 1-epsilon. Se escoge cualquier otro algoritmo con probabilidad
# epsilon.
class epsilonGreedy(Bandit):
    def __init__(self,ratings,epsilon=0.1):
        super().__init__(ratings)
        self.epsilon = epsilon

    # Devuelve el índice del brazo seleccionado según el algoritmo
    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        t0 = time()
        accuracy = np.array(list(map(lambda x: x.accuracy(),availableArms)))
        t1 = time()

        # Hallamos el brazo de máxima precisión y lo separamos de choices
        best = availableArms[np.argmax(accuracy)]
        availableArms = np.delete(availableArms,np.argmax(accuracy))
        t2 = time()
        # for i in range(len(self.times)):
        #     self.times[i] += t[i+1] - t[i]

        self.times[1] += t1 - t0
        self.times[2] += t2 - t1

        # Elegimos la mejor opción o una de las otras aleatoriamente según el criterio
        # explicado antes
        number = random.random()
        if(number < self.epsilon):
            return np.random.choice(availableArms)
        else:
            return best

# Bandido que escoge un brazo al azar
class randomBandit(Bandit):
    def __init__(self,ratings):
        super().__init__(ratings)

    # Sabiendo los items que ya ha visto el usuario, se devuleve otro aleatoriamente
    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        return np.random.choice(availableArms)

# Bandido que escoge un brazo al azar en función de una distribución beta
# dependiente del número de aciertos y errores de cada brazo
class thompsonSampling(Bandit):
    def __init__(self,ratings,alpha=0.01,beta=0.01):
        super().__init__(ratings)
        self.alpha = alpha
        self.beta = beta

    # Devuelve el índice del brazo seleccionado según el algoritmo
    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        numbers = np.array([np.random.beta(arm.hits+self.alpha,arm.fails+self.beta) for arm in availableArms])
        return availableArms[np.argmax(numbers)]
