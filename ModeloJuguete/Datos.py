import pandas as pd
import numpy as np

def isNominal(x):
  #El tipo que guarda de las Strings es Object
    if x.kind == 'O':
        return True
    else:
        return False

def encodeAtribute(datos):
    keys = sorted(list(set(datos)),key=str.lower)
    return {k:i for i, k in enumerate(keys)}


class Datos:

    def __init__(self, nombreFichero, nominal=True):
        #Inicialización datos
        self.df = pd.read_csv(nombreFichero, header=0)
        self.datos = self.df.values
        self.cols = self.df.columns.values

        #Inicialización nominalAtributos
        self.types = self.df.dtypes
        types = self.types.array
        self.nominalAtributos = [isNominal(x) for x in types]

        if nominal:
            #Inicialización diccionario
            self.diccionario = [encodeAtribute(self.datos[:,i]) if val else {} for i, val in enumerate(self.nominalAtributos)]

            for i in range(np.shape(self.datos)[1]):
                if self.nominalAtributos[i]:
                    self.datos[:,i] = [self.diccionario[i].get(val) for val in self.datos[:,i]]

    def extraeDatos(self, idx):
        return self.datos[ idx ]

    def extraeCols(self, names):
        cols = []
        for i, col in enumerate(self.cols):
            if col in names:
                cols.append(i)

        return self.datos[:,cols]
