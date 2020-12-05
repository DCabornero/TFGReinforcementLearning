from abc import abstractmethod


import time

# Cada uno de los Arms tendrá un sistema de recomendación que permita
# dar un elemento a recomendar para un cierto usuario siguiendo un cierto algoritmo.
# El trainSet debe ser dado en formato de matriz numPy.
class Arm:
    def __init__(self):
        self.hits = 0
        self.fails = 0
        self.misses = 0

    def accuracy(self):
        if self.hits + self.fails + self.misses== 0:
            return 0
        else:
            return self.hits/(self.hits+self.fails+self.misses)

    # Dado un usuario recomienda un cierto item
    @abstractmethod
    def rec_item(self,user):
        pass

class ArmItem(Arm):
    def __init__(self,item):
        super().__init__()
        self.item = item

    def rec_item(self):
        return self.item
