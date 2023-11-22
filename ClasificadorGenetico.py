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

    def __elitism():
        #TODO: coge los individuos mejores predictores, y los saca de la poblacion.
        pass

    def __fitness(self, xdata: np.ndarray , ydata: np.ndarray, individual: list, diccionario: dict):
        
        #TODO: por cada una de las lineas de xdata, se predice el resultado y se añade a una lista, se
        #      devolverá la tasa de error del individuo
        pass

    def __populate(self):
        """
            Genera una poblacion de individuos con sus propias reglas
        """        
        self.population = []
        for _ in self.numPopulation:
            individial = []

            n_rules = random.randint(1,self.numRules)
            for _ in range(len(n_rules)):
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

        for _ in range(hight):
            fitness_list = []
            for ind in self.population:
                fitness_list.append(self.__fitness(x_train, y_train, ind, diccionario))

        pass

    def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
        pass

