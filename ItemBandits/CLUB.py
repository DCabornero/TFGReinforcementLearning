from abstract.Bandit import Bandit
from abstract.Context import Context
from abstract.Graph import Graph

import numpy as np
import pandas as pd
import operator

class CLUB(Bandit):
    # alpha: parámtero de exploración
    # beta: parámtero de eliminación de enlace
    def __init__(self,alpha,beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return 'CLUB'

    # Al pasarle el fichero tags, se inicializa la clase contexto
    def read_tags_csv(self,tags,userName='userId',itemName='movieId',tagName='tag', timeName='timestamp'):
        self.context = Context()
        self.context.read_csv(tags)

    # Se añade a diccionarios la matriz y el vector correspondiente a cada usuario y a cada cluster
    def add_itemArms(self):
        super().add_itemArms()
        self.graph = Graph(self.listUsers)

        A = np.eye(self.context.numTags)
        b = np.zeros(self.context.numTags).reshape(-1,1)
        # Diccionarios del usuario
        self.user_matrix_dict = {user:np.copy(A) for user in self.listUsers}
        self.user_matrix_inv_dict = {user:np.copy(A) for user in self.listUsers}
        self.user_b_dict = {user:np.copy(b) for user in self.listUsers} #Vector b
        self.user_w_dict = {user:np.copy(b) for user in self.listUsers} #Vector w
        self.user_epochs_dict = {user:0 for user in self.listUsers}
        self.user_conf_dict = {user:self.beta for user in self.listUsers}
        # Diccionarios de los brazos: cada contexto y cada cota
        self.cont_dict = {item:self.context.context(item).reshape(-1,1) for item in self.listItems}
        # Diccionarios de los clusters
        self.cluster_matrix_dict = {0:np.copy(A)}
        self.cluster_matrix_inv_dict = {0:np.copy(A)}
        self.cluster_vector_dict = {0:np.copy(b)} # Vector w

        self.epochs = 0


    def select_arm(self,viewed):
        availableArms = self.available_arms(viewed)
        probs = []
        cluster = self.graph.find_cluster(self.target)
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

    # Acutaliza las matrices y vectores relativas a un cierto cluster
    # TODO
    def update_cluster(self,cluster):
        users = self.graph.dictClusters[cluster]
        matr = np.eye(self.context.numTags)
        b = np.zeros(self.context.numTags).reshape(-1,1)
        id = np.eye(self.context.numTags)
        for user in users:
            matr += self.user_matrix_dict[user] - id
            b += self.user_b_dict[user]
        matr_inv = np.linalg.inv(matr)
        w = np.matmul(matr_inv,b)
        # Sustitución en diccionarios
        self.cluster_matrix_dict[cluster] = matr
        self.cluster_matrix_inv_dict[cluster] = matr_inv
        self.cluster_vector_dict[cluster] = np.matmul(matr_inv,b)

    # Comunica a los diccionarios que un cierto usuario ha sido escogido otra época
    def update_epoch(self,user):
        self.user_epochs_dict[user] += 1
        t = self.user_epochs_dict[user]
        self.user_conf_dict[user] = self.beta*np.sqrt((1+np.log(1+t))/(1+t))
        self.epochs += 1


    def update_preference(self,item,reward):
        # Obtención de datos
        x = self.cont_dict[item]
        matr = self.user_matrix_dict[self.target]
        b = self.user_b_dict[self.target]
        # Actualización de datos
        matr += np.matmul(x,x.T)
        matr_inv = np.linalg.inv(matr)
        # Probar a reescalar con [-1,1]
        b += reward*x
        w = np.matmul(matr_inv,b)
        # Se meten a la tabla
        self.user_matrix_dict[self.target] = matr
        self.user_b_dict[self.target] = b
        self.user_w_dict[self.target] = w
        self.user_matrix_inv_dict[self.target] = matr_inv
        # Si es necesario, reforma del grafo # TODO
        confidence = self.user_conf_dict[self.target]
        for us in self.graph.connected_users(self.target):
            norm = np.linalg.norm(w-self.user_w_dict[us])
            conf_sum = confidence + self.user_conf_dict[us]
            if conf_sum < norm:
                cl = self.graph.disconnect(self.target,us)
                if cl:
                    self.update_cluster(cl[0])
                    self.update_cluster(cl[1])
        # Aumento de época
        self.update_epoch(self.target)

    def item_hit(self,item):
        super().item_hit(item)
        self.update_preference(item,1)

    def item_fail(self,item):
        super().item_fail(item)
        self.update_preference(item,0)

    def item_miss(self,item):
        super().item_miss(item)
        self.update_preference(item,0)
