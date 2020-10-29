from abc import abstractmethod

import numpy as np
import pandas as pd

from Datos import Datos
import Arm

from sklearn.model_selection import train_test_split

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

    # # Se añade un brazo con algoritmo Naive-Bayes item-based
    def add_itemNB(self):
        cols = ['movieId','genres']
        moviesNB =  self.movies.extraeCols(cols)
        self.arms.append(Arm.ArmItemNB(moviesNB))

    # Depende del algoritmo de selección concreto
    @abstractmethod
    def select_arm(self):
        pass

    # Queda run epoch
