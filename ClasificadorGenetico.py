import numpy as np
import random
from Clasificador import Clasificador
import math
from typing import Any
import pandas as pd

class ClasificadorGenetico(Clasificador):
    def __init__(self, numPopulation: int =50,  epoches: int = 50, numRules: int = 5, \
                elit_prob: float = 0.05, roule_prob: float = 0.02, mutation_prob: float = 0.05, \
                bitmut_prob: float = 0.15) -> Any:
        self.numPopulation = numPopulation
        self.epoches = epoches
        self.numRules = numRules
        self.elit_prob = elit_prob
        self.roule_prob = roule_prob
        self.mutation_prob = mutation_prob
        self.bitmut_prob = bitmut_prob

    def __elitism(self, fitness_list: list) -> list:
        #TODO: coge los individuos mejores predictores, y los saca de la poblacion.
        num_elits = int(len(fitness_list)*self.elit_prob)
        elite_list = []

        for _ in num_elits:
            elit = max(fitness_list)
            elit_index = fitness_list.index(elit)
            elite_list.append(elit_index)
            fitness_list.remove(elit)

        return elite_list

    def __predict(self, data: np.ndarray, individual: list, diccionario: dict):
        
        dict_list = list(diccionario.items())[:-1]  # Obtenemos una lista del contenido
                                                    # del diccionario excepto el ultimo elemento

        predicted_classes = []
        for rule in individual:
            print("="*20, f"{rule}", "="*20)
            idx = 0
            aux = 1
            for j in range(len(dict_list)):
                print("="*10, f"{j}", "="*10)
                n_subdict = len(dict_list[j])
                data = data.astype(int)
                rule_np = np.array(rule)
                matches = np.bitwise_and(data[idx:idx+n_subdict], rule_np[idx:idx+n_subdict])
                if sum(matches) == 0:
                    print("Nop")
                    aux = 0
                    break
                print("Sip")
                idx += n_subdict
            if aux == 1:
                predicted_classes.append(rule[-1])

        print(f"{sum(predicted_classes)} ? {len(predicted_classes)}")

        if sum(predicted_classes) > len(predicted_classes)/2:
            print(f"{sum(predicted_classes)} > {len(predicted_classes)}")
            print("a")
            return 1
        elif sum(predicted_classes) < len(predicted_classes)/2:
            print(f"{sum(predicted_classes)} < {len(predicted_classes)}")
            print("b")
            return 0

        return None

    def __fitness(self, xdata: np.ndarray , ydata: np.ndarray, diccionario: dict):
        
        #TODO: por cada una de las lineas de xdata, se predice el resultado y se añade a una lista, se
        #      devolverá la tasa de error del individuo
        fitness_list = []
        for ind in self.population:
            correct = 0
            for i in range(xdata.shape[0]):
                data = xdata[i,:]

                pred = self.__predict(data, ind, diccionario)

                if pred == ydata[i]:
                    correct += 1
            fitness_list.append(float(correct)/xdata.shape[0])
        return fitness_list

    def __populate(self):
        """
            Genera una poblacion de individuos con sus propias reglas
        """        
        self.population = []
        for _ in range(self.numPopulation):
            individial = []

            n_rules = random.randint(1,self.numRules)
            for _ in range(n_rules):
                rule = []
                # Creamos una regla que no sea ni todo 0s ni todo 1s
                while sum(rule) == 0 or sum(rule) == len(rule):
                    rule = random.choices([0,1], k=self.rules_length)

                individial.append(rule)
            self.population.append(individial)

    def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos, diccionario):
        x_train = datosTrain.iloc[:, :-1].values
        y_train = datosTrain.iloc[:, -1].values

        hight, width = x_train.shape

        self.rules_length = width+1

        self.__populate()

        for _ in range(self.epoches):

            fitness_list = self.__fitness(x_train, y_train, diccionario)

            #self.__elitism(fitness_list)
        print(fitness_list)

    def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
        pass

