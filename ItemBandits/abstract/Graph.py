import numpy as np

class Graph:
    def __init__(self,listUsers):
        self.listUsers = listUsers
        # Se comienza con un grafo conexo. Esta matriz simétrica indica si el nodo
        # i-ésimo está conectado al nodo j-ésimo o no (mediante unos y ceros)
        self.boolMatrix = np.ones((len(listUsers),len(listUsers)))
        # Correspondencia entre los índices de la matriz y el user correspondiente
        self.dictUsers = {user:i for i,user in enumerate(listUsers)}
        # Diccionario que indica en qué componente conexa se encuentra cada user
        self.dictClusters = {0:self.listUsers}
        self.numClusters = 1

    # Desconexión del dos users. Devuelve True si se ha generado un nuevo cluster,
    # false en caso contrario.
    def disconnect(self,user1,user2):
        # Actualizamos la matriz de conexiones
        index1 = self.dictUsers[user1]
        index2 = self.dictUsers[user2]
        self.boolMatrix[index1,index2] = 0
        self.boolMatrix[index2,index1] = 0
        # Contador de nodos visitados
        i = 0
        # Lista de nodos pendientes de visitar y nodos visitados.
        pend = [user1]
        # Condición de parada: ya no hay más nodos pendientes de visitar, por lo
        # que pend es la lista completa de nodos conexos a user1
        while i < len(pend):
            # Hallamos el user a explorar y añadimos los vecinos a la lista de
            # pendientes
            index = self.dictUsers[pend[i]]
            vecinos = self.listUsers[self.boolMatrix[index]==1]
            for nodo in vecinos:
                # Si se ha encontrado un camino entre la separación, sigue siendo
                # una única componente conexa.
                if nodo == user2:
                    return None
                # Si el nodo no está en la lista de pendientes por visitar, se añade.
                elif not nodo in pend:
                    pend.append(nodo)
            i += 1
        # Si no se ha encontrado user2, hay dos componentes conexas: una con los
        # valores encontrados en "pend" y otra con los no encontrados. Se crea
        # un nuevo cluster con los valores de pend
        pend = np.array(pend)
        cluster = self.find_cluster(user1)
        new_cluster = np.setdiff1d(self.dictClusters[cluster],pend)
        self.dictClusters[cluster] = pend
        self.dictClusters[self.numClusters] = new_cluster
        # self.dictClusters.update({user:self.numClusters for user in pend})
        self.numClusters += 1
        # Se devuelven los clusters que se han separado
        return [cluster, self.numClusters-1]

    # Devuelve el cluster perteneciente a un user
    def find_cluster(self,user):
        for i in range(self.numClusters):
            if user in self.dictClusters[i]:
                return i

    # Devuelve los users que están conectados en el grafo a un cierto user
    def connected_users(self,user):
        index = self.dictUsers[user]
        return self.listUsers[self.boolMatrix[index]==1]
