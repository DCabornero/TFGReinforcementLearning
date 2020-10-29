from abc import abstractmethod
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.naive_bayes import MultinomialNB

# Cada uno de los Arms tendrá un sistema de recomendación que permita
# dar un elemento a recomendar para un cierto usuario siguiendo un cierto algoritmo.
# El trainSet debe ser dado en formato de matriz numPy.
class Arm:
    def __init__(self):
        self.hits = 0
        self.fails = 0

    # Dado un usuario recomienda un cierto item
    @abstractmethod
    def rec_item(self,user):
        pass

    # Indica a un cierto brazo cual es el trainSet sobre el que debe entrenarse
    @abstractmethod
    def initSet(self,trainSet):
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
        return [self.user_avg(trainSet,user) for user in listUsers]


# Sistema de recomendación kNN, donde la similitud entre vecinos queda definida
# por el coeficiente de correlación de Pearson. El trainSet a pasar debe contener tres
# columnas: userID, itemID y el rating (en este orden).
class ArmkNN(Arm):
    def __init__(self,k):
        self.k = k

    def initSet(self,trainSet):
        self.trainSet = trainSet

    # Intersección de items comunes. Devuelve una lista de dos elementos: cada
    # uno es el trainSet que queda cuando solo tenemos a los elementos de la intersección
    def intersect_items(self, trainSet1, trainSet2):
        mask1 = np.isin(trainSet1[:,1],trainSet2[:,1])
        mask2 = np.isin(trainSet2[:,1],trainSet1[:,1])
        return [trainSet1[mask1,:],trainSet2[mask2,:]]

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
    def rec_item(self,user):
        userSet = self.user_set(self.trainSet,user)

        # Lista de usuarios sin el usuario target
        listUsers = np.unique(self.trainSet[:,0])
        listUsers = listUsers[listUsers != user]

        # Lista de items evaluados por el target
        userItems = np.unique(self.trainSet[self.trainSet[:,0] == user,1])
        # Lista de items no recomendados al target
        mask = np.logical_not(np.isin(self.trainSet[:,1],userItems))
        listItems = np.unique(self.trainSet[mask,1])

        usersInfo = {}
        avgUser = self.user_avg(self.trainSet,user)

        for u in listUsers:
            userSet2 = self.user_set(self.trainSet,u)
            avg2 = self.user_avg(self.trainSet,u)
            usersInfo[u] = [avg2,self.Pearson(userSet,userSet2,avgUser,avg2)]

        # Nos vamos quedando con el item con mejor rating
        item = -1
        rating = 0
        for i in listItems:
            newRating = self.rate(user,avgUser,i,self.trainSet,usersInfo)
            if newRating > rating:
                item = i
                rating = newRating
        return item



class ArmNB(Arm):
    # El trainSet está compuesto por tres columnas: userId, itemId, rating
    def initSet(self,trainSet):
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
        self.items = np.unique(genres[:,0])
        self.num_items = len(self.items)
        self.dataset = self.items.copy()
        self.listCols = []
        self.add_genres(genres)
        self.clf = MultinomialNB()

    def initSet(self,trainSet):
        self.trainSet = self.modify_trainSet(trainSet)

    # Comprueba si un tag está ya introducido. Si no lo está se crea una columna, si
    # lo está se introduce la estadísitca.
    def check_genre(self,tag,item):
        row = list(self.items).index(item)
        if tag in self.listCols:
            col = self.listCols.index(tag) + 1
        else:
            self.listCols.append(tag)
            col = len(self.listCols)
            self.dataset = np.column_stack((self.dataset,np.zeros(self.num_items)))
        self.dataset[row,col] = 1

    # Dado un dataset de items y sus géneros separados por '|', devuelve la tabla
    # con cada genero distinto siendo una columno
    def add_genres(self,genres):
        for row in genres:
            for tag in row[1].split('|'):
                self.check_genre(tag,row[0])

    # Dado un dataset de items y sus tags obtenemos la prolongación de la tabla que
    # devuleve cada tag distinto en una columna
    # def add_tags(self,tags,tPop):
    #     # Nos quedamos con los tags que se han puesto en más de tag popularity películas
    #     listTags, count = np.unique(tags[:,1],return_counts=True)
    #     print(np.shape(listTags))
    #     listTags = listTags[count >= 5] #Utilizando los params de unique posiblemente se puedan hacer cosillas
    #     print(np.shape(listTags))
    #
    #     tagsMatrix = np.zeros(self.num_items,len(listTags))
    #     # np.apply_along_axis(,0,tagsMatrix)
    #     self.listCols += listTags

    # Modifica los ratings por 1 y -1 en función de si las valoraciones están por encima o debajo de la media
    def modify_trainSet(self,trainSet):
        # Diccionario cuyas claves son los usuarios y los valores su media
        list_users = np.unique(trainSet[:,0])
        dictAvg = {u:self.user_avg(trainSet,u) for u in list_users}

        func = lambda x: 1 if dictAvg.get(x[0]) <= x[2] else -1
        new_ratings = np.apply_along_axis(func,1,trainSet)

        trainSet[:,2] = new_ratings
        return trainSet

    # Recomendación de un item a un cierto usuario target
    def rec_item(self,user):
        # Nuestro trainSet en esta recomendación son los items ya valorados por el usuario
        # El resto formarán el testSet
        # Comenzamos hallando los ratings binarios (-1,1) para los items visualizados por el target
        mask = self.trainSet[:,0] == user
        itemRatings = (self.trainSet[mask,:])[:,[1,2]]
        itemSet = itemRatings[:,0]

        # Fabricamos nuestros conjuntos train y test
        func = lambda x: True if x[0] in itemSet else False
        mask = np.apply_along_axis(func,1,self.dataset) #NO HAY DATOS REPETIDO EN DATASET

        train = self.dataset[mask]
        test = self.dataset[np.logical_not(mask)]

        # Sabiendo que train y itemRatings tienen los mismos items, los ordenamos y
        # obtendremos el conjunto de entrenamiento y sus resultados
        orderTrain = np.argsort(train,axis=0)[:,0]
        X_train = (train[orderTrain])[:,1:]
        orderRatings = np.argsort(itemRatings,axis=0)[:,0]
        y_train = (itemRatings[orderRatings])[:,1]

        X_test = test[:,1:]

        # Entrenamos con scipy y obtenemos las predicciones
        self.clf.fit(X_train,y_train)
        preds = self.clf.predict_proba(X_test)

        bestPred = np.argmax(preds[:,1])
        return test[bestPred,0]
