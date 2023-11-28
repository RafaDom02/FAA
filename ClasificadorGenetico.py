import copy
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

    def __isTrue(self, ruleIndividual, ruleLine) -> bool:
        for i in range(len(ruleIndividual)-1):
            if ruleIndividual[i] != 0:
                if ruleIndividual[i] == ruleLine[i]:
                    return False
        return True

    def __parents_selection(self, fitness_list):
        #TODO: selecciona los padres
        pass


    def __elitism(self, fitness_list: list) -> list:
        #TODO: coge los individuos mejores predictores, y los saca de la poblacion. REVISAR POR SI ACASO
        num_elits = math.ceil(len(fitness_list)*self.elit_prob)
        elite_list = []

        fitness_list_copy = copy.deepcopy(fitness_list)

        for _ in num_elits:
            elit = max(fitness_list_copy)
            elit_index = fitness_list_copy.index(elit)
            elite_list.append(elit_index)
            fitness_list_copy.remove(elit)

        return elite_list

    def __predict(self, line: np.ndarray, individual: list):
        #TODO: REVISAR y ARREGLAR, solo funciona con datos en binario
        # Esto solo funciona por ahora con xor ya que cada linea del xor ya esta en binario
        predicted_classes = []
        predicted_rules = []
        for rule in individual:
            if self.__isTrue(rule, line):
                predicted_rules.append(rule[-1])

        if not predicted_rules:
            return None
        
        counts = np.bincount(predicted_rules)
        predicted_classes.append(np.argmax(counts))

        #print(f"{sum(predicted_classes)} ? {len(predicted_classes)}")

        if sum(predicted_classes) > len(predicted_classes)/2:
            #print(f"{sum(predicted_classes)} > {len(predicted_classes)}")
            return 1
        elif sum(predicted_classes) < len(predicted_classes)/2:
            #print(f"{sum(predicted_classes)} < {len(predicted_classes)}")
            return 0

        return None

    def __fitness(self, xdata: np.ndarray , ydata: np.ndarray):
        
        #TODO: por cada una de las lineas de xdata, se predice el resultado y se añade a una lista, se
        #      devolverá la tasa de error del individuo
        fitness_list = []
        for ind in self.population:
            correct = 0
            for i in range(xdata.shape[0]):
                data = xdata[i,:]

                pred = self.__predict(data, ind)

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

        _, width = x_train.shape

        self.rules_length = width+1

        self.__populate()

        for _ in range(self.epoches):

            fitness_list = self.__fitness(x_train, y_train)

            elit_index_list = self.__elitism(fitness_list)
            print(fitness_list)
            
            # Obtenemos los padres

            #Crossover: cruce a partir de los padres crear nuevas soluciones

            #Mutations: los padres reciben mutaciones

            #La poblacion pasa a ser una nueva 

    def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
        pass

