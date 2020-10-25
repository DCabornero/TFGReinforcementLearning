from abc import abstractmethod
import numpy as np

# Cada uno de los Arms tendrá un sistema de recomendación que permita
# dar un elemento a recomendar para un cierto usuario siguiendo un cierto algoritmo.
# El trainSet debe ser dado en formato de matriz numPy.
class Bandit:
    @abstractmethod
    def recc_film(self,trainSet,user):
        pass

# Sistema de recomendación kNN, donde la similitud entre vecinos queda definida
# por el coeficiente de correlación de Pearson. El trainSet a pasar debe contener tres
# columnas: userID, itemID y el rating (en este orden).
class kNNBandit(Bandit):
    # # Cálculo de la media de las valoraciones de un usuario
    # def user_avg(trainSet, user):
