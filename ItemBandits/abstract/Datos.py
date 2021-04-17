import pandas as pd
import numpy as np

class Datos:

    def __init__(self, nombreFichero):
        #Inicialización datos
        self.df = pd.read_csv(nombreFichero, header=0)
        self.datos = self.df.values
        self.cols = self.df.columns.values

    def extraeDatos(self, idx):
        return self.datos[ idx ]

    def extraeCols(self, names):
        cols = []
        for i, col in enumerate(self.cols):
            if col in names:
                cols.append(i)

        return self.datos[:,cols]
