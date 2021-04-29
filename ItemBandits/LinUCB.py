from abstract.Bandit import Bandit
from abstract.Context import Context

import numpy as np
import pandas as pd
import operator

class LinUCB(Bandit):
    contextual = True
    def __init__(self,alpha=0.1):
        super().__init__()
        self.alpha = alpha

    def __str__(self):
        return 'LinUCB'

    # Se añade a diccionarios la matriz, el vector y el contexto correspondiente a cada item
    def add_itemArms(self):
        super().add_itemArms()
        A = np.eye(self.context.numTags)
        b = np.zeros(self.context.numTags).reshape(-1,1)
        self.matrix_dict = {item:np.copy(A) for item in self.listItems}
        self.matrix_inv_dict = {item:np.copy(A) for item in self.listItems}
        self.vector_dict = {item:np.copy(b) for item in self.listItems}
        self.cont_dict = {item:self.context.context(item).reshape(-1,1) for item in self.listItems}

    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        probs = []
        for item in availableArms:
            # x: contexto
            x = self.cont_dict[item]
            # Matriz y vector asociados al item
            inv_A = self.matrix_inv_dict[item]
            b = self.vector_dict[item]
            theta = np.matmul(np.matmul(inv_A,b).T,x)
            probs.append(theta[0,0] + self.alpha*np.sqrt(np.matmul(x.T,np.matmul(inv_A,x)))[0,0])
        choice = availableArms[np.argmax(probs)]
        return choice

    def update_preference(self,item,reward):
        # Obtención de datos
        x = self.cont_dict[item]
        A = self.matrix_dict[item]
        b = self.vector_dict[item]
        # Actualización de datos
        A += np.matmul(x,x.T)
        b += reward*x
        # Se meten a la tabla
        self.matrix_dict[item] = A
        self.vector_dict[item] = b
        self.matrix_inv_dict[item] = np.linalg.inv(A)

    def item_hit(self,item):
        super().item_hit(item)
        self.update_preference(item,1)

    def item_fail(self,item):
        super().item_fail(item)
        self.update_preference(item,0)

    def item_miss(self,item):
        super().item_miss(item)
        self.update_preference(item,0)
