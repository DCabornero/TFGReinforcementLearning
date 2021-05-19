import matplotlib.pyplot as plt
import numpy as np

class Analysis():
    # path: ruta donde se van a guardar las imágenes
    # name: nombre de las imágenes
    def __init__(self,path=None,name=''):
        self.path = path
        self.name = name
    # Ejecuta varios bandits con las épocas indicadas. El usuario es responsable
    # de introducir un contexto en los bandits si fuera necesario.
    # bandits: array de los bandits a ejecutar
    # epochs: número de épocas que va a ejecutarse cada bandit
    def execute(self,bandits,epochs,**kwargs):
        self.bandits = bandits
        if 'gini_history' in kwargs.keys():
            self.gini_history = kwargs['gini_history']
        else:
            self.gini_history = False

        if 'time_history' in kwargs.keys():
            self.time_history = kwargs['time_history']
        else:
            self.time_history = False

        for bandit in bandits:
            print('Ejecutando',str(bandit)+'...')
            bandit.add_itemArms()
            bandit.run_epoch(epochs,**kwargs)

    # Devuelve una gráfica con la evolución del recall acumulado de los bandits
    def recall(self):
        plt.figure()
        for bandit in self.bandits:
            plt.plot(np.arange(len(bandit.recall)),bandit.recall,label=str(bandit))
        plt.xlabel('Épocas')
        plt.ylabel('Recall acumulado')
        plt.title('Evolución del recall acumulado')
        plt.legend()
        if self.path:
            plt.savefig(self.path+'/'+self.name+'recall.png')
        plt.show()

    # Devuelve una gráfica con el tiempo de ejecución de cada bandit.
    def time(self):
        plt.figure()
        fig, ax = plt.subplots()
        names = [str(bandit) for bandit in self.bandits]
        times = [bandit.times for bandit in self.bandits]
        ax.barh(np.arange(len(self.bandits)),times)
        ax.set_yticks(np.arange(len(self.bandits)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Tiempo (segundos)')
        ax.set_title('Comparación de tiempos')
        if self.path:
            plt.savefig(self.path+'/'+self.name+'times.png')
        plt.show()
        if self.time_history:
            histories = [bandit.time_history for bandit in self.bandits]
            plt.xlabel('Épocas')
            plt.ylabel('Tiempo transcurrido (s)')
            for i,history in enumerate(histories):
                plt.plot(np.arange(len(history)),history,label=str(self.bandits[i]))
            plt.legend()
            if self.path:
                plt.savefig(self.path+'/'+self.name+'times_hist.png')
            plt.show()

    # Devuelve una gráfica con el tiempo de ejecución de cada bandit.
    def gini(self):
        plt.figure()
        fig, ax = plt.subplots()
        names = [str(bandit) for bandit in self.bandits]
        ginis = [bandit.gini() for bandit in self.bandits]
        ax.barh(np.arange(len(self.bandits)),ginis)
        ax.set_yticks(np.arange(len(self.bandits)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Coeficiente de Gini')
        ax.set_title('Comparación de coeficientes de Gini')
        if self.path:
            plt.savefig(self.path+'/'+self.name+'gini.png')
        plt.show()
        if self.gini_history:
            histories = [bandit.gini_history for bandit in self.bandits]
            plt.xlabel('Épocas')
            plt.ylabel('Coeficiente de Gini')
            for i,history in enumerate(histories):
                plt.plot(np.arange(len(history)),history,label=str(self.bandits[i]))
            plt.legend()
            if self.path:
                plt.savefig(self.path+'/'+self.name+'gini_hist.png')
            plt.show()
