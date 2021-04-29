import matplotlib.pyplot as plt
import numpy as np

class Analysis():
    # Ejecuta varios bandits con las épocas indicadas. El usuario es responsable
    # de introducir un contexto en los bandits si fuera necesario.
    # bandits: array de los bandits a ejecutar
    # epochs: número de épocas que va a ejecutarse cada bandit
    def execute(self,bandits,epochs,trainSize=0):
        self.bandits = bandits
        for bandit in bandits:
            print('Ejecutando',str(bandit)+'...')
            bandit.add_itemArms()
            bandit.run_epoch(epochs,trainSize=trainSize)

    # Devuelve una gráfica con la evolución del recall acumulado de los bandits
    def recall(self):
        plt.figure()
        for bandit in self.bandits:
            plt.plot(np.arange(len(bandit.recall)),bandit.recall,label=str(bandit))
        plt.xlabel('Épocas')
        plt.ylabel('Recall acumulado')
        plt.set_title('Evolución del recall acumulado')
        plt.legend()
        plt.show()

    # Devuelve una gráfica con el tiempo de ejecución de cada bandit.
    def time(self):
        plt.figure()
        fig, ax = plt.subplots()
        names = [str(bandit) for bandit in self.bandits]
        times = [bandit.times for bandit in self.bandits]
        ax.barh(np.arange(len(bandits)),times)
        ax.set_yticks(np.arange(len(self.bandits)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Tiempo (segundos)')
        ax.set_title('Comparación de tiempos')
        plt.show()
