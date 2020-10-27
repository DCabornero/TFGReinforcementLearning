from abc import abstractmethod
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize

# Cada uno de los Arms tendrá un sistema de recomendación que permita
# dar un elemento a recomendar para un cierto usuario siguiendo un cierto algoritmo.
# El trainSet debe ser dado en formato de matriz numPy.
class Arm:
    def __init__(self):
        self.hits = 0
        self.fails = 0

    @abstractmethod
    def recc_item(self,trainSet,user):
        pass

# Sistema de recomendación kNN, donde la similitud entre vecinos queda definida
# por el coeficiente de correlación de Pearson. El trainSet a pasar debe contener tres
# columnas: userID, itemID y el rating (en este orden).
class ArmkNN(Arm):
    def __init__(self,k):
        self.k = k

    # Submatriz que solo contiene datos de un cierto usuario
    def user_set(self, trainSet, user):
        mask = trainSet[:,0] == user
        return trainSet[mask]

    # Intersección de items comunes. Devuelve una lista de dos elementos: cada
    # uno es el trainSet que queda cuando solo tenemos a los elementos de la intersección
    def intersect_items(self, trainSet1, trainSet2):
        mask1 = np.isin(trainSet1[:,1],trainSet2[:,1])
        mask2 = np.isin(trainSet2[:,1],trainSet1[:,1])
        return [trainSet1[mask1,:],trainSet2[mask2,:]]


    # Cálculo de la media de las valoraciones de un usuario
    def user_avg(self, trainSet, user):
        userRatings = self.user_set(trainSet, user)[:,2]
        if(len(userRatings) == 0):
            return 2.5
        return np.mean(userRatings)

    # Cálculo de la media de valoraciones de los usuarios
    def user_avgs(self, trainSet):
        listUsers = np.unique(trainSet[:,0])
        return [self.user_avg(trainSet,user) for user in listUsers]

    # Cálculo de la similitud entre dos usuarios mediante el coeficiente de correlación de
    # Pearson
    def Pearson(self,set1,set2,avg1,avg2):
        inters1, inters2 = self.intersect_items(set1,set2)
        if np.shape(inters1)[0] == 0 or np.shape(inters2)[0] == 0:
            return 0
        dif1 = inters1[:,2]-avg1
        dif2 = inters2[:,2]-avg2
        numer = np.sum(np.multiply(dif1, dif2))
        den1 = np.linalg.norm(dif1)
        den2 = np.linalg.norm(dif2)
        if den1 == 0 or den2 == 0:
            return 0
        return numer/(den1*den2)

    # Devuelve los usuarios que hayan valorado una cierta película
    def get_users(self, trainSet, item):
        mask = trainSet[:,1] == item
        return trainSet[mask,:]

    # Elimina los resultados relativos a un cierto usuario
    def remove_user(self, trainSet, user):
        mask = trainSet[:,0] != user
        return trainSet[mask,:]

    # Devuelve el rating predicho para un cierto item por un usuario
    # usersInfo es un diccionario donde la clave es el usuario y el valor que contiene
    # es un array con la media de ratings y la similitud con el objetivo
    def rate(self, user, avguser, item, trainSet, usersInfo):
        # Calculamos el conjunto de usuarios distintos del target que han visto el item
        users = self.get_users(trainSet,item)
        users = self.remove_user(users,user)

        # Si no hay suficientes vecinos, es una mala predicción
        if np.shape(users)[0] < self.k:
            return 0

        # Para estos usuarios hacemos un array que contenga el usuario, su rating,
        # su media y su similitud con el target
        data = np.concatenate((users[:,[0,2]],[usersInfo.get(user) for user in users[:,0]]),axis=1)

        # Ordenamos de menor a mayor las similitudes y nos quedamos con las k últimas
        data = data[np.argsort(data[:,1]),:]
        data = data[-self.k:,:]

        # Hallamos numerador y denominador
        numer = np.sum(np.multiply(data[:,3],data[:,1]-data[:,2]))
        den = np.sum(np.abs(data[:,]))

        return avguser + numer/den

    # Función principal: recomienda un item a un cierto usuario
    def rec_item(self,trainSet,user):
        userSet = self.user_set(trainSet,user)

        # Lista de usuarios sin el usuario target
        listUsers = np.unique(trainSet[:,0])
        listUsers = listUsers[listUsers != user]

        # Lista de items evaluados por el target
        userItems = np.unique(trainSet[trainSet[:,0] == user,1])
        # Lista de items no recomendados al target
        mask = np.logical_not(np.isin(trainSet[:,1],userItems))
        listItems = np.unique(trainSet[mask,1])

        usersInfo = {}
        avgUser = self.user_avg(trainSet,user)

        for u in listUsers:
            userSet2 = self.user_set(trainSet,u)
            avg2 = self.user_avg(trainSet,u)
            usersInfo[u] = [avg2,self.Pearson(userSet,userSet2,avgUser,avg2)]

        # Nos vamos quedando con el item con mejor rating
        item = -1
        rating = 0
        for i in listItems:
            newRating = self.rate(user,avgUser,i,trainSet,usersInfo)
            if newRating > rating:
                item = i
                rating = newRating
        return item

class ArmNB(Arm):
    # El trainSet está compuesto por tres columnas: userId, itemId, rating
    def __init__(self,trainSet):
        self.trainSet = trainSet
        self.ratings = np.unique(trainSet[:,2])
        self.num_ratings = len(self.ratings)
        self.items = np.unique(trainSet[:,1])
        self.num_items = len(self.items)
        self.users = np.unique(trainSet[:,0])
        self.num_users = len(self.users)

    # Se calculan los priores (ratings) para un cierto item
    def priores(self, item):
        priors = np.zeros(self.num_ratings)
        for i, rt in enumerate(self.ratings):
            mask = self.trainSet[:,1] == item
            priors[i] = np.count_nonzero(self.trainSet[mask,2] == rt)
        return priors/sum(priors)

    # Un usario puntúa un item (1) puntuado por el target y ha valorado (2) el item deseado.
    # Esta función devuelve ejemplos con dos columnas: el primero es el item 1 y el segundo
    # es la valoración 2.
    # item es el item 1 y itemsTarget es la lista de valoraciones de todos los usuarios teniendo
    # solo en cuenta los items valorados por el target
    def tablaCondicional(self,item,itemsTarget):
        # Nos quedamos con los usuarios que han valorado el item y sus ratings
        mask = self.trainSet[:,1] == item
        users = self.trainSet[mask]
        users = self.trainSet[:,[0,2]]

        usersList = list(users[:,0])

        # Filtramos itemsTarget a los usuarios que han valorado el item
        func = lambda x: x[0] in usersList
        mask = np.apply_along_axis(func,1,itemsTarget)
        finalItems = itemsTarget[mask]

        # Cambiamos el rating del item por el rating del item (1) del usuario
        # Esta lambda obtiene el rating del item (1) del usuario
        lmb = lambda x: users[usersList.index(x[0]),1]
        ratings = np.apply_along_axis(lmb,1,finalItems)
        finalItems[:,2] = ratings
        return finalItems[:,[1,2]]

    # De la tabla de resultados anterior obtenemos una tabla de frecuencias normalizada
    # La dimensión de la tabla es numItems x num_ratings, donde numItems es el número de
    # items distintos que haya en tablaCondicional. La normalización consiste en que las columnas suman 1.
    def tablaFrecs(self,tCond):
        listItems = list(np.unique(tCond[:,0]))
        listRatings = list(self.ratings)
        # Normalización de Lagrange por defecto
        frecs = np.zeros((len(listItems),self.num_ratings))

        unique, counts = np.unique(tCond,return_counts=True,axis=1)
        # LE DA LA VUELTA AL PUTO ARRAY
        print(tCond)
        print(unique)
        counts += 1
        for i, u in enumerate(unique):
            print(u)
            frecs[listItems.index(u[0]),listRatings.index(u[1])] = counts[i]

        return normalize(frecs, axis=1, norm='l1')
