from abc import abstractmethod
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB

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
        if self.hits + self.fails == 0:
            return 0
        else:
            return self.hits/(self.hits+self.fails)

    # Dado un usuario recomienda un cierto item
    @abstractmethod
    def rec_item(self,user):
        pass

    # Indica a un cierto brazo cual es el trainSet sobre el que debe entrenarse
    @abstractmethod
    def initSet(self,trainSet):
        pass

    # Añade un ejemplo al trainSet
    @abstractmethod
    def add_sample(self,sample):
        pass

    # Añade una mala valoración a un ejemplo del que desconocemos el rating
    @abstractmethod
    def add_bad_sample(self,user,item):
        pass

    # Submatriz que solo contiene datos de un cierto usuario
    def user_set(self, trainSet, user):
        mask = trainSet[:,0] == user
        return trainSet[mask]

    # Cálculo de la media de las valoraciones de un usuario
    def user_avg(self, trainSet, user):
        userRatings = self.user_set(trainSet, user)[:,2]
        if(len(userRatings) == 0):
            return 2.5
        return np.mean(userRatings)

    # Cálculo de la media de valoraciones de los usuarios
    def user_avgs(self, trainSet):
        listUsers = np.unique(trainSet[:,0])
        return np.array(list(map(lambda x: self.user_avg(trainSet,x),listUsers)))


# Sistema de recomendación kNN, donde la similitud entre vecinos queda definida
# por el coeficiente de correlación de Pearson. El trainSet a pasar debe contener tres
# columnas: userID, itemID y el rating (en este orden).
class ArmkNN(Arm):
    def __init__(self,k):
        super().__init__()
        self.k = k

    def initSet(self,trainSet):
        self.trainSet = trainSet.copy()
        avgs = self.user_avgs(trainSet)
        self.userInfo = pd.DataFrame(np.transpose(avgs), columns=['avg'], index=np.unique(trainSet[:,0]))

    def add_sample(self,sample):
        self.trainSet = np.vstack((self.trainSet,sample))
        self.userInfo.at[sample[0],'avg'] = self.user_avg(self.trainSet,sample[0])

    def add_bad_sample(self,user,item):
        self.add_sample([user,item,1])

    # Intersección de items comunes. Devuelve una lista de dos elementos: cada
    # uno es el trainSet que queda cuando solo tenemos a los elementos de la intersección
    def intersect_items(self, trainSet1, trainSet2):
        mask1 = np.isin(trainSet1[:,1],trainSet2[:,1])
        mask2 = np.isin(trainSet2[:,1],trainSet1[:,1])
        return [trainSet1[mask1],trainSet2[mask2]]

    # Cálculo de la similitud entre dos usuarios mediante el coeficiente de correlación de
    # Pearson
    def Pearson(self,set1,set2,avg1,avg2):
        inters1, inters2 = self.intersect_items(set1,set2)
        if np.shape(inters1)[0] == 0 or np.shape(inters2)[0] == 0:
            return 0
        if np.shape(inters1)[0] != np.shape(inters2)[0]:
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
    def get_users(self, item):
        mask = self.trainSet[:,1] == item
        return self.trainSet[mask,:]

    # Elimina los resultados relativos a un cierto usuario
    def remove_user(self, set, user):
        mask = set[:,0] != user
        return set[mask,:]

    # Devuelve el rating predicho para un cierto item por un usuario target
    # usersInfo es un DataFrame cuyas columnas son la media de valoraciones y la similitud
    # con el target de cada usuario
    def rate(self, user, item, usersInfo): # RECUERDO: HACER UN USERSINFO
        # Calculamos el conjunto de usuarios distintos del target que han visto el item
        users = self.get_users(item)
        users = self.remove_user(users,user)

        # Si no hay suficientes vecinos, es una mala predicción
        if np.shape(users)[0] < self.k:
            return 0

        # Hallamos los k usuarios con mayor similitud
        candidates = usersInfo.loc[users[:,0]]
        nearest = candidates.sort_values(by=['sim'],ascending=False).head(self.k)

        # Añadimos al DataFrame el rating de dicho item y convertimos los datos a array numpy
        nearest = nearest.join(pd.DataFrame(np.transpose(users[:,2]), columns=['rating'], index=users[:,0]))
        data = nearest.to_numpy()

        # Hallamos numerador y denominador
        numer = np.sum(np.multiply(data[:,1],data[:,2]-data[:,0]))
        den = np.sum(np.abs(data[:,1]))

        if den == 0:
            return 0

        res = usersInfo.at[user,'avg'] + numer/den

        return res

    # Función principal: recomienda un item a un cierto usuario
    def rec_item(self,user):
        userSet = self.user_set(self.trainSet,user)

        # Lista de usuarios sin el usuario target
        listUsers = self.userInfo.index.to_numpy()
        listUsers = listUsers[listUsers != user]

        # Lista de items evaluados por el target
        userItems = np.unique(self.trainSet[self.trainSet[:,0] == user,1])
        # Lista de items no recomendados al target
        mask = np.logical_not(np.isin(self.trainSet[:,1],userItems))
        listItems = np.unique(self.trainSet[mask,1])

        usersInfo = self.userInfo.copy()

        # Cálculo de similitudes
        simsCol = \
            np.array([self.Pearson(userSet,self.user_set(self.trainSet,u),usersInfo.at[user,'avg'],usersInfo.at[u,'avg']) for u in listUsers])
        sims = pd.DataFrame(np.transpose(simsCol),columns=['sim'],index=listUsers)
        usersInfo = usersInfo.join(sims)

        # Nos vamos quedando con el item con mejor rating
        item = -1
        rating = 0
        for i in listItems:
            newRating = self.rate(user,i,usersInfo)
            if newRating > rating:
                item = i
                rating = newRating
        return item



class ArmNB(Arm):
    # El trainSet está compuesto por tres columnas: userId, itemId, rating
    def initSet(self,trainSet):
        self.trainSet = trainSet.copy()
        self.ratings = np.unique(trainSet[:,2])
        self.num_ratings = len(self.ratings)
        self.items = np.unique(trainSet[:,1])
        self.num_items = len(self.items)
        self.users = np.unique(trainSet[:,0])
        self.num_users = len(self.users)

    def add_sample(self,sample):
        self.initSet(np.vstack((self.trainSet,sample)))

    def add_bad_sample(self,user,item):
        self.add_sample([user,item,1])

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

    # De la tabla de resultados anterior obtenemos una tabla de frecuencias.
    # La dimensión de la tabla es numItems x num_ratings, donde numItems es el número de
    # items distintos que haya en tablaCondicional. Se realiza la normalización de Lagrange.
    def tablaFrecs(self,tCond):
        listItems = list(np.unique(tCond[:,0]))
        listRatings = list(self.ratings)
        # Normalización de Lagrange por defecto
        frecs = np.ones((len(listItems),self.num_ratings))

        # Ponemos el rating como la parte decimal del item, ya que unique no trabaja
        # bien comparando arrays (cambia de sitio los elementos)
        newtCond = tCond[:,0] + tCond[:,1]*0.1

        unique, counts = np.unique(newtCond,return_counts=True)
        for i, u in enumerate(unique):
            item = np.floor(u)
            rt = np.around((u-item)*10,decimals=1)
            frecs[listItems.index(item),listRatings.index(rt)] += counts[i]
        return frecs

    # Devuelve un rating esperado de un cierto item. itemsTarget es la lista de valoraciones de todos los usuarios teniendo
    # solo en cuenta los items valorados por el target
    def exp_rating(self, itemsTarget, item):
        tC = self.tablaCondicional(item,itemsTarget)
        frecs = self.tablaFrecs(tC)
        priores = self.priores(item)
        # Array de los productos de las probabilidades condicionales para cada rating posible
        prodCond = np.prod(frecs,axis=0)
        # Multiplicamos cada uno por su prior correspondiente y normalizamos
        prod = np.multiply(prodCond,priores)
        norm = np.linalg.norm(prod,ord=1)
        prod /= norm
        # Lo obtenido son los pesos de cada rating, con el producto escalar con los ratings
        # obtenemos la valoración esperada
        return np.dot(self.ratings,prod)

    # Recomendación para un cierto target
    def rec_item(self,target):
        # Calculamos los items valorados por el usuario y después itemsTarget será
        # cualquier ejemplo que contenga alguno de esos items
        mask = self.trainSet[:,0] == target
        items = list(np.unique(self.trainSet[mask,1]))
        mask = np.apply_along_axis(lambda x: x[1] in items,1,self.trainSet)
        itemsTarget = self.trainSet[mask]

        # Items que no ha valorado el target
        # mask = self.items[lambda x: x not in items]
        mask = [x not in items for x in self.trainSet[:,1]]
        notRated = self.trainSet[mask,1]

        func = lambda x: exp_rating(itemsTarget,x)
        ratings = map(func,notRated)


        bestIndex = np.argmax(ratings)
        bestItem = notRated[bestIndex]
        return bestItem



# Algoritmo Naive-Bayes item-based
class ArmItemNB(Arm):
    #genres: matriz numpy cuyas columnas son un item y géneros separados por barras
    #tags: matriz numpy con columnas: item, tag
    #tag_popularity: solo se tienen en cuenta los tags que hayan aparecido más veces que este número
    #trainSet: trainSet habitual con usuario, item y rating
    def __init__(self,genres):
        super().__init__()
        self.items = np.unique(genres[:,0])
        self.num_items = len(self.items)
        self.generos = pd.DataFrame(index=self.items, columns=[])
        self.add_genres(genres)
        self.clf = MultinomialNB()

    def initSet(self,trainSet):
        self.trainSet = self.modify_trainSet(trainSet)

    def add_sample(self,sample):
        if sample[2] <= self.dictAvg.get(sample[0]):
            self.trainSet = np.vstack((self.trainSet,[sample[0],sample[1],-1]))
        else:
            self.trainSet = np.vstack((self.trainSet,[sample[0],sample[1],1]))


    def add_bad_sample(self,user,item):
        self.trainSet = np.vstack((self.trainSet,[user,item,-1]))

    # Comprueba si un tag está ya introducido. Si no lo está se crea una columna, si
    # lo está se introduce la estadísitca.
    def check_genre(self,tag,item):
        self.generos.at[item,tag] = 1

    # Dado un dataset de items y sus géneros separados por '|', devuelve la tabla
    # con cada genero distinto siendo una columno
    def add_genres(self,genres):
        for row in genres:
            for tag in row[1].split('|'):
                self.check_genre(tag,row[0])
        self.generos.fillna(0, inplace=True)

    # Modifica los ratings por 1 y -1 en función de si las valoraciones están por encima o debajo de la media
    def modify_trainSet(self,trainSet):
        # Diccionario cuyas claves son los usuarios y los valores su media
        trainSetCopy = trainSet.copy()
        list_users = np.unique(trainSetCopy[:,0])
        self.dictAvg = {u:self.user_avg(trainSetCopy,u) for u in list_users}

        func = lambda x: 1 if self.dictAvg.get(x[0]) <= x[2] else -1
        new_ratings = np.apply_along_axis(func,1,trainSetCopy)

        trainSetCopy[:,2] = new_ratings
        return trainSetCopy

    # Recomendación de un item a un cierto usuario target
    def rec_item(self,user):
        # Nuestro trainSet en esta recomendación son los items ya valorados por el usuario
        # El resto formarán el testSet
        # Comenzamos hallando los ratings binarios (-1,1) para los items visualizados por el target
        mask = self.trainSet[:,0] == user
        itemRatings = (self.trainSet[mask,:])[:,[1,2]]
        itemSet = itemRatings[:,0]

        ratingsDf = pd.DataFrame(index=itemSet,columns=[])
        ratingsDf['ratings'] = itemRatings[:,1]

        # Fabricamos nuestros conjuntos train y test, donde utilizamos para train todos los items valorados
        # por el target
        maskTrain = self.generos.index.isin(itemSet)
        train = self.generos.loc[itemSet]
        maskTest = np.logical_not(maskTrain)
        test = self.generos.loc[maskTest]

        # Sabiendo que train y itemRatings tienen los mismos items, los ordenamos y
        # obtendremos el conjunto de entrenamiento y sus resultados
        train = train.join(ratingsDf,how='inner')

        # Entrenamos con scipy y obtenemos las predicciones
        npTrain = train.to_numpy()
        self.clf.fit(npTrain[:,:-1],npTrain[:,-1])
        preds = self.clf.predict_proba(test.to_numpy())

        bestPred = np.argmax(preds[:,1])
        return test.index[bestPred]
