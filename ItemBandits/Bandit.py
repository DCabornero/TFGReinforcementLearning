from abc import abstractmethod

import numpy as np
import pandas as pd

from Datos import Datos

from sklearn.model_selection import train_test_split

import random

import matplotlib.pyplot as plt

from time import time

# Cada uno de los Arms tendrá un sistema de recomendación que permita
# dar un elemento a recomendar para un cierto usuario siguiendo un cierto algoritmo.
# El trainSet debe ser dado en formato de matriz numPy.
class Bandit:
    # userName: nombre de la columna que contiene los userID
    # itemName: nombre de la columna que contiene los itemID
    # ratingName: nombre de la columna que contiene los ratings
    # timeName: timestamp de cada valoracion
    def __init__(self,userName='userId',itemName='movieId',ratingName='rating', timeName='timestamp'):
        self.arms = None
        self.results = None
        self.times = np.zeros((3))
        self.names = {'user':userName,
                      'item':itemName,
                      'rating': ratingName,
                      'time': timeName}

    # ratings: CSV que contiene todos los ejemplos posibles con mínimo tres columnas:
    # - ID del usuario (default: userId)
    # - ID del item (default: itemId)
    # - valorción que ha dado dicho usuario a dicho item (default: rating)
    def read_csv(self,ratings):
        self.ratings = Datos(ratings)
        self.listUsers = np.unique(self.ratings.extraeCols([self.names['user']]))
        self.listItems = np.unique(self.ratings.extraeCols([self.names['item']]))

        # Obtención del rating medio (medio camino entre máximo y mínimo)
        ratings = np.unique(self.ratings.extraeCols(self.names['rating']))
        self.avgRating = np.mean(ratings)

    def add_itemArms(self):
        empty = np.zeros((len(self.listItems)))
        self.arms = pd.DataFrame({'Hits': empty, 'Fails': empty, 'Misses': empty}, index=self.listItems)

    # Crea un diccionario que tiene a los usuarios por key y un array de sus
    # items ya valorados como valor
    def get_rated(self,trainSet):
        viewed = {}
        empty = len(trainSet) == 0
        for u in self.listUsers:
            if empty:
                itemsRated = np.array([])
            else:
                itemsRated = trainSet[trainSet[:,0] == u,1]
            viewed[u] = itemsRated
        return viewed

    # Devuelve la lista de los items no valorados sabiendo cuales se han valorado ya
    def available_arms(self,viewed):
        keys = self.arms.index
        availableKeys = np.setdiff1d(keys,viewed,assume_unique=True)

        # arr = keys[np.isin(keys,viewed,assume_unique=True,invert=True)]

        return availableKeys

    # Métodos que incrementan el contador de hits, fails y misses. Susceptibles de sobreescribirse
    # en clases heredadas
    def item_miss(self,item):
        self.arms.loc[item,'Misses'] += 1

    def item_hit(self,item):
        self.arms.loc[item,'Hits'] += 1

    def item_fail(self,item):
        self.arms.loc[item,'Fails'] += 1

    # Depende del algoritmo de selección concreto
    @abstractmethod
    def select_arm(self,viewed):
        pass

    # Corre un cierto número de épocas con un algoritmo especificado.
    # Devuelve un array con dos listas: la primera son las épocas y la segunda
    # el recall relativo en cada época (que corresponde con la gráfica mostrada)
    # Si shuffle está a False, el conjunto de entrenamiento serán los elementos más antiguos
    def run_epoch(self,epochs=500,trainSize=0.1,shuffle=True):
        index = 0
        epoch = 0
        numhits = 0

        # Ordenación por timestamp
        cols = [self.names[x] for x in ['user','item','rating']]
        ordered_index = np.argsort(self.ratings.extraeCols([self.names['time']])[:,0])
        ord_ratings = self.ratings.extraeCols(cols)[ordered_index]
        if trainSize > 0:
            train, test = train_test_split(ord_ratings, train_size=trainSize, shuffle=shuffle)
        else:
            train = np.array([])
            test = ord_ratings


        viewed = self.get_rated(train)
        self.add_itemArms()
        results = [[],[]]

        # Número de hits por descubrir
        totalhits = np.shape(test[test[:,2]>=self.avgRating])[0]

        numUsers = len(self.listUsers)
        random.shuffle(self.listUsers)

        # Comienzan a correr las épocas
        while epoch < epochs:
            target = self.listUsers[index]
            t0 = time()
            item = self.select_arm(viewed[target])
            t1 = time()
            self.times[0] += t1-t0

            # Comprobamos si tenemos la recomendación del item en el testSet
            mask = np.logical_and(test[:,0] == target,test[:,1] == item)

            # Si hemos encontrado un resultado, lo valoramos como hit o fail
            if(np.count_nonzero(mask) > 0):
                # epoch += 1
                test, hit = test[np.logical_not(mask)], test[mask]
                if hit[0,2] >= self.avgRating:
                    self.item_hit(item)
                    numhits += 1
                else:
                    self.item_fail(item)
            else:
                self.item_miss(item)
            viewed[target] = np.append(viewed[target],[item])

            results[0].append(epoch)
            results[1].append(numhits/totalhits)

            index += 1
            # Cuando agotamos los usuarios los barajamos y volvemos a empezar
            if index == numUsers:
                random.shuffle(self.listUsers)
                index = 0

            epoch += 1
        print(self.times)
        self.results = results

        return results

    def plot_results(self):
        if self.results:
            plt.plot(self.results[0],self.results[1])
        else:
            print('No hay resultados para graficar.')
