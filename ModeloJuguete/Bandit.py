from abc import abstractmethod

import numpy as np
import pandas as pd

from Datos import Datos
import Arm

from sklearn.model_selection import train_test_split

import random

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
    def __init__(self,ratings,movies):
        self.ratings = Datos(ratings)
        self.movies = Datos(movies,nominal=False)
        self.arms = []

    # Se añade un brazo con algoritmo kNN user-based
    def add_knnArm(self,k):
        self.arms.append(Arm.ArmkNN(k))

    # Se añade un brazo con algoritmo Naive-Bayes user-based
    def add_NB(self):
        self.arms.append(Arm.ArmNB())

    # Se añade un brazo con algoritmo Naive-Bayes item-based
    def add_itemNB(self):
        cols = ['movieId','genres']
        moviesNB =  self.movies.extraeCols(cols)
        self.arms.append(Arm.ArmItemNB(moviesNB))

    # Depende del algoritmo de selección concreto
    @abstractmethod
    def select_arm(self):
        pass

    # Queda run epoch
    def run_epoch(self,epochs=500,trainSize=0.1):
        index = 0
        epoch = 0
        cols = ['userId','movieId','rating']
        train, test = train_test_split(self.ratings.extraeCols(cols), train_size=trainSize)
        for arm in self.arms:
            arm.initSet(train)

        listUsers = np.unique(train[:,0])
        numUsers = len(listUsers)
        random.shuffle(listUsers)

        # Comienzan a correr las épocas
        while epoch < epochs:
            target = listUsers[index]
            arm = self.select_arm()

            # Se recomiendo un item
            item = arm.rec_item(target)

            # Comprobamos si tenemos la recomendación del item en el testSet
            mask = np.logical_and(test[:,0] == target,test[:,1] == item)

            # Si hemos encontrado un resultado, lo valoramos como hit o fail
            if(np.count_nonzero(mask) > 0):
                # epoch += 1
                test, hit = test[np.logical_not(mask)], test[mask]
                for a in self.arms:
                    # Los trainSet se duplican con cada arm: PUNTO A MEJORAR
                    a.add_sample(hit[0])
                # De momento, el umbral de valoración hit/fail es 3
                if hit [0,2] >= 2:
                    arm.hits += 1
                else:
                    arm.fails += 1
            else:
                arm.misses += 1
                for a in self.arms:
                    a.add_bad_sample(target,item)

            index += 1
            # Cuando agotamos los usuarios los barajamos y volvemos a empezar
            if index == numUsers:
                random.shuffle(listUsers)
                index = 0

            epoch += 1

# Algoritmo epsilon-greedy. Se elige el algoritmo con mayor tasa de hits
# con probabilidad 1-epsilon. Se escoge cualquier otro algoritmo con probabilidad
# epsilon.
class epsilonGreedy(Bandit):
    def __init__(self,ratings,movies,epsilon=0.1):
        super().__init__(ratings,movies)
        self.epsilon = epsilon

    # Devuelve el índice del brazo seleccionado según el algoritmo
    def select_arm(self):
        choices = [i for i in range(len(self.arms))]
        accuracy = [arm.accuracy() for arm in self.arms]

        # Hallamos el brazo de máxima precisión y lo separamos de choices
        best = choices.pop(np.argmax(accuracy))

        # Elegimos la mejor opción o una de las otras aleatoriamente según el criterio
        # explicado antes
        number = random.random()
        if(number < self.epsilon):
            return self.arms[random.choice(choices)]
        else:
            return self.arms[best]

class randomBandit(Bandit):
    def __init__(self,ratings,movies):
        super().__init__(ratings,movies)

    # Devuelve el índice del brazo seleccionado según el algoritmo
    def select_arm(self):
        choices = [i for i in range(len(self.arms))]
        return self.arms[random.choice(choices)]
