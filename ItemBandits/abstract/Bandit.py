from abc import abstractmethod

import numpy as np
import pandas as pd

from Datos import Datos
from Context import Context

from sklearn.model_selection import train_test_split

import random

import matplotlib.pyplot as plt

from time import time
# Posiblemente lo quite en la versión final
# try:
#     get_ipython().__class__.__name__
#     from tqdm.notebook import tqdm
# except:
#     from tqdm import tqdm

# Cada uno de los Arms tendrá un sistema de recomendación que permita
# dar un elemento a recomendar para un cierto usuario siguiendo un cierto algoritmo.
# El trainSet debe ser dado en formato de matriz numPy.
class Bandit:
    # De predeterminado un algoritmo es contextual
    contextual = False
    # Modos de shuffle que se ofrecen
    shuffle_mode = ['random','balanced','round_robin']

    def __init__(self):
        self.arms = None
        self.results = None
        self.times = 0

    # ratings: CSV que contiene todos los ejemplos posibles con mínimo tres columnas:
    # - ID del usuario (default: userId)
    # - ID del item (default: itemId)
    # - valorción que ha dado dicho usuario a dicho item (default: rating)
    # userName: nombre de la columna que contiene los userID
    # itemName: nombre de la columna que contiene los itemID
    # ratingName: nombre de la columna que contiene los ratings
    def read_csv(self,ratings,userName='userId',itemName='movieId',ratingName='rating'):
        self.ratings = Datos(ratings)
        self.names = {'user':userName,
                      'item':itemName,
                      'rating': ratingName}
        self.listUsers = np.unique(self.ratings.extraeCols([self.names['user']]))
        self.listItems = np.unique(self.ratings.extraeCols([self.names['item']]))

        # Obtención del rating medio (medio camino entre máximo y mínimo)
        ratings = np.unique(self.ratings.extraeCols(self.names['rating']))
        self.avgRating = np.mean(ratings)

    # Al pasarle el fichero tags, se inicializa la clase contexto. Solo es necesario si se requiere un contexto.
    def read_tags_csv(self,tags,userName='userId',itemName='movieId',tagName='tag'):
        self.context = Context()
        self.context.read_csv(tags)

    # Crea el dataframe que va a guardar todas las características necesarias de cada item
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
        self.rewards.append(self.miss_value)

    def item_hit(self,item):
        self.arms.loc[item,'Hits'] += 1
        self.rewards.append(1)

    def item_fail(self,item):
        self.arms.loc[item,'Fails'] += 1
        self.rewards.append(0)

    # Depende del algoritmo de selección concreto
    @abstractmethod
    def select_arm(self,viewed,user=None):
        pass

    # Corre un cierto número de épocas con un algoritmo especificado.
    # Devuelve un array con dos listas: la primera son las épocas y la segunda
    # el recall relativo en cada época (que corresponde con la gráfica mostrada)
    # - epochs: número de épocas que hace el algoritmo
    # - miss_value: valor que se le da a la recompensa cuando el resultado no existe (recomendado 0)
    # - miss_not_epoch: si es True, no se pasa de época cuando no se encuentra dato
    # - shuffle: tiene 3 modos de barajar a los usuarios:
    #       * balanced: recorre la lista de usuarios, y casa vez que la acaba hace shuffle (aleatorio balanceado)
    #       * random: siempre escoge un usuario al azar por época (aleatorio no balanceado)
    #       * round_robin: solo barajea los usuarios al principio. Va recorriendo la lista en orden.
    # - gini_history: guarda el histórico de los coeficientes de Gini cada época
    # - time_history: guarda la evolución del tiempo cada época
    def run_epoch(self,epochs=500,miss_value=0,miss_not_epoch=False,shuffle='balanced',
                  gini_history=False, time_history=False):
        if shuffle not in self.shuffle_mode:
            print('Error: no existe el modo', shuffle)
            return
        index = 0
        epoch = 0
        numhits = 0
        self.rewards = []

        self.miss_value = miss_value
        if gini_history:
            self.gini_history = []
        if time_history:
            self.time_history = []


        t0 = time()
        # Ordenación por timestamp
        cols = [self.names[x] for x in ['user','item','rating']]
        # ordered_index = np.argsort(self.ratings.extraeCols([self.names['time']])[:,0])
        # ord_ratings = self.ratings.extraeCols(cols)[ordered_index]
        # if trainSize > 0:
        #     train, test = train_test_split(ord_ratings, train_size=trainSize, shuffle=shuffle)
        # else:
        #     train = np.array([])
        #     test = ord_ratings
        # test = self.ratings.extraeCols(cols)[ordered_index]
        test = self.ratings.extraeCols(cols)

        # viewed = self.get_rated(train)
        viewed = {u:[] for u in self.listUsers}
        self.add_itemArms()
        recall = []
        rewards = []

        # Número de hits por descubrir
        totalhits = np.shape(test[test[:,2]>=self.avgRating])[0]

        numUsers = len(self.listUsers)
        random.shuffle(self.listUsers)

        # Comienzan a correr las épocas
        while epoch < epochs:
            # bar.update()
            self.target = self.listUsers[index]

            item = self.select_arm(viewed[self.target])

            # Comprobamos si tenemos la recomendación del item en el testSet
            mask = np.logical_and(test[:,0] == self.target,test[:,1] == item)

            # Si hemos encontrado un resultado, lo valoramos como hit o fail
            if(np.count_nonzero(mask) > 0):
                # epoch += 1
                test, hit = test[np.logical_not(mask)], test[mask]
                if hit[0,2] >= self.avgRating:
                    self.item_hit(item)
                    numhits += 1
                else:
                    self.item_fail(item)
                miss = False
            else:
                miss = True
                self.item_miss(item)
            viewed[self.target] = np.append(viewed[self.target],[item])

            # Se añaden los datos históricos necesarios
            if not (miss_not_epoch and miss):
                recall.append(numhits/totalhits)
            if gini_history and not (miss_not_epoch and miss):
                self.gini_history.append(self.gini())
            if time_history and not (miss_not_epoch and miss):
                self.time_history.append(time()-t0)


            index += 1
            # Barjeo de usuarios
            if index == numUsers:
                if shuffle == 'balanced':
                    random.shuffle(self.listUsers)
                    index = 0
                elif shuffle == 'round_robin':
                    index = 0
            if shuffle == 'random':
                random.shuffle(self.listUsers)
                index = 0
            # Pase de época
            if not (miss_not_epoch and miss):
                epoch += 1

        t1 = time()
        self.times = t1-t0
        self.recall = recall

    # Una vez hallados los resultados, halla el coeficiente de Gini
    def gini(self):
        sorted = np.sort(np.sum(self.arms[['Hits','Misses','Fails']].to_numpy(),axis=1),kind='mergesort')
        n = len(sorted)
        indices = np.arange(n) + 1
        return np.sum((2*indices - n - 1) * sorted) / (n * np.sum(sorted))


    def plot_results(self):
        if self.results:
            plt.plot(self.results[0],self.results[1])
        else:
            print('No hay resultados para graficar.')
