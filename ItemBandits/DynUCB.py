from abstract.Bandit import Bandit
from abstract.Context import Context

import numpy as np
import pandas as pd
import operator

class DynUCB(Bandit):
    def __init__(self,conf=0.05,clusters=10):
        super().__init__()
        self.alpha = 1+ np.sqrt(np.log(2/conf)/2)
        self.numClusters = clusters

    # Al pasarle el fichero tags, se inicializa la clase contexto
    def read_tags_csv(self,tags,userName='userId',itemName='movieId',tagName='tag', timeName='timestamp'):
        self.context = Context()
        self.context.read_csv(tags)

    # Se añade a diccionarios la matriz y el vector correspondiente a cada usuario y a cada cluster
    def add_itemArms(self):
        super().add_itemArms()
        A = np.eye(self.context.numTags)
        b = np.zeros(self.context.numTags).reshape(-1,1)
        # Diccionarios del usuario
        self.user_matrix_dict = {user:np.copy(A) for user in self.listUsers}
        self.user_matrix_inv_dict = {user:np.copy(A) for user in self.listUsers}
        self.user_vector_dict = {user:np.copy(b) for user in self.listUsers} #Vector b
        self.cluster_dict = {user:np.random.randint(self.numClusters) for user in self.listUsers}
        # Diccionario de los brazos con los contextos
        self.cont_dict = {item:self.context.context(item).reshape(-1,1) for item in self.listItems}
        # Diccionarios de los clusters
        self.cluster_matrix_dict = {cluster:np.copy(A) for cluster in np.arange(self.numClusters)}
        self.cluster_matrix_inv_dict = {cluster:np.copy(A) for cluster in np.arange(self.numClusters)}
        self.cluster_vector_dict = {cluster:np.copy(b) for cluster in np.arange(self.numClusters)} # Vector w

        self.epochs = 0


    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        probs = []
        cluster = self.cluster_dict[self.target]
        # Vector y matriz del cluster necesarios
        inv_matr = self.cluster_matrix_inv_dict[cluster]
        w_vector = self.cluster_vector_dict[cluster]
        for item in availableArms:
            # x: contexto
            x = self.cont_dict[item]
            probs.append(np.matmul(w_vector.T,x)[0,0] \
            + self.alpha*np.sqrt(np.matmul(x.T,np.matmul(inv_matr,x))*np.log(self.epochs+1))[0,0])
        choice = availableArms[np.argmax(probs)]
        return choice

    # Dado un cierto cluster devuelve la lista de usuarios del mismo.
    def users_in_cluster(self,cluster):
        listClusters = np.array(list(self.cluster_dict.values())) == cluster
        return np.array(list(self.cluster_dict.keys()))[listClusters]

    # Acutaliza las matrices y vectores relativas a un cierto cluster
    def update_cluster(self,cluster):
        users = [k for k in self.listUsers if self.cluster_dict[k]==cluster]
        matr = np.eye(self.context.numTags)
        b = np.zeros(self.context.numTags).reshape(-1,1)
        id = np.eye(self.context.numTags)
        for user in users:
            matr += self.user_matrix_dict[user] - id
            b += self.user_vector_dict[user]
        matr_inv = np.linalg.inv(matr)
        # Sustitución en diccionarios
        self.cluster_matrix_dict[cluster] = matr
        self.cluster_matrix_inv_dict[cluster] = matr_inv
        self.cluster_vector_dict[cluster] = np.matmul(matr_inv,b)


    def update_preference(self,item,reward):
        # Obtención de datos
        x = self.cont_dict[item]
        matr = self.user_matrix_dict[self.target]
        b = self.user_vector_dict[self.target]
        # Actualización de datos
        matr += np.matmul(x,x.T)
        matr_inv = np.linalg.inv(matr)
        # Probar a reescalar con [-1,1]
        b += reward*x
        w = np.matmul(matr_inv,b)
        # Se meten a la tabla
        self.user_matrix_dict[self.target] = matr
        self.user_vector_dict[self.target] = b
        self.user_matrix_inv_dict[self.target] = matr_inv
        # Reasignación del usuario en el cluster:
        prev_cluster = self.cluster_dict[self.target]
        listNorms = np.array([np.linalg.norm(self.cluster_vector_dict[cl]-w) for cl in range(self.numClusters)])
        new_clusters = np.where(listNorms==listNorms.min())[0]
        new_cluster = np.random.choice(new_clusters)
        # Actualización de clusters
        if prev_cluster != new_cluster:
            self.cluster_dict[self.target] = new_cluster
            self.update_cluster(new_cluster)
        self.update_cluster(prev_cluster)

        # Aumento de época
        self.epochs += 1

    def item_hit(self,item):
        super().item_hit(item)
        self.update_preference(item,1)

    def item_fail(self,item):
        super().item_fail(item)
        self.update_preference(item,0)

    def item_miss(self,item):
        super().item_miss(item)
        self.update_preference(item,0)
