import copy
import numpy as np
import random
from Clasificador import Clasificador
import math
from typing import Any
import pandas as pd

class ClasificadorGenetico(Clasificador):
    def __init__(self, numPopulation: int = 50,  epoches: int = 50, numRules: int = 5, elit_prob: float = 0.05, \
                cross_prob: float = 0.02, mutation_prob: float = 0.05, bitmut_prob: float = 0.15, \
                math_related_prediction: bool = True, debug_rules: bool = False, debug_fitness: bool = False) -> Any:
        self.numPopulation = numPopulation
        self.epoches = epoches
        self.numRules = numRules
        self.elit_prob = elit_prob
        self.cross_prob = cross_prob
        self.mutation_prob = mutation_prob
        self.bitmut_prob = bitmut_prob
        self.math_related_prediction = math_related_prediction
        self.debug_rules = debug_rules
        self.debug_fitness = debug_fitness

    def __isTrue(self, ruleIndividual, ruleLine) -> bool:
        """
        Author:
            Rafael Dominguez Saez

        Comprueba si una regla de un individuo con la regla de una linea del dataset

        Args:
            ruleIndividual (_type_): _description_
            ruleLine (_type_): _description_

        Returns:
            bool: _description_
        """
        for i in range(len(ruleIndividual)-2):
            if ruleIndividual[i] != 0:
                if ruleIndividual[i] == ruleLine[i]:
                    return False
        return True

    def __parents_selection(self, fitness_list):
        """
        Author:
            Rafael Dominguez Saez

        Hacemos la selección de los padres haciendo el metodo de la "ruleta".

        Args:
            fitness_list (_type_): Lista de los fitnesses por individuo

        Returns:
            list: Lista de los individuos seleccionados para ser los padres 
        """        

        # La elite pasa directamente a la siguiente generacion
        elite_size = math.ceil(self.elit_prob * len(fitness_list))

        # Calcula la suma total de los valores de fitness
        total_fitness = sum(fitness_list)

        # Calcula los pesos relativos de cada individuo
        weights = [fitness / total_fitness for fitness in fitness_list]

        # Selecciona dos padres usando np.random.choice con los pesos calculados
        selected_parents = random.choices(self.population, weights, k=len(self.population) - elite_size)

        return selected_parents

    def __crossover(self, parents):
        """
        Raliza sobre los padres el crossover para generar los descendientes si se cumple la condicion
        de cross_prob, con igual de condiciones de realizar el intra o inter crossover

        Args:
            parents (list): Lista de los padres

        Returns:
            list: descendientes
        """

        descendents = []
        
        # Pair in tuples all parents in a random way
        random.shuffle(parents)
        if len(parents) % 2 != 0: #Para el caso de que los padres sean impares
            soltero = parents.pop()
            descendents.append(soltero)

        PairsOfParents = list(zip(*[iter(parents)]*2))

        for pareja in PairsOfParents:
            padre1, padre2 = pareja
            #get random number from 0 to 1 and check if its less than cross_prob
            if random.random() < self.cross_prob:
                #intra crossover (cambian cachitos de reglas)
                if random.random() < 0.5:
                    #get random number from 0 to rules length
                    cross_point = random.randint(0, len(padre1[0])-1)
                    elected_rule = random.randint(0, min(len(padre1), len(padre2))-1)
                    #split a random rule from padre1 and padre2
                    aux_rule = padre1[elected_rule][:cross_point] + padre2[elected_rule][cross_point:]
                    padre1[elected_rule] = aux_rule
                    descendents.append(padre1)

                    aux_rule = padre2[elected_rule][:cross_point] + padre1[elected_rule][cross_point:]
                    padre2[elected_rule] = aux_rule
                    descendents.append(padre2)

                #inter crossover (reglas completas)
                else:
                    #get random number from 0 to numRules
                    cross_point = random.randint(0, self.numRules-1)

                    descendents.append(padre1[:cross_point] + padre2[cross_point:])
                    descendents.append(padre2[:cross_point] + padre1[cross_point:])
            else:
                descendents.append(padre1)
                descendents.append(padre2)

        return descendents
    
    def __bitflip_mutation(self, parents, diccionario):
        """
        Author:
            Pablo Sanchez

        Hace la mutacion de un bit en una de las reglas de un individuo

        Args:
            parents (list): lista de los padres
            diccionario (dict): diccionario del dataset

        Returns:
            list: inidividuos tras tener una mutacion
        """
        
        for individuo in parents:
            for regla in individuo:
                indexBegin = 0
                indexEnd = 0
                for key in diccionario.keys():
                    lenkeys = len(list(diccionario[key].keys()))
                    indexEnd += lenkeys

                    if random.random() < self.bitmut_prob:
                        i = random.randint(indexBegin, indexEnd - 1)
                        for j in range(indexBegin, indexEnd):
                            regla[j] = 0
                        regla[i] = 1

                    indexBegin = indexEnd 
        return parents
        
    def __rule_mutation(self, parents, diccionario):
        """
        Author:
            Pablo Sanchez

        Hace la mutación añadiendo o eliminando una regla a un individuo

        Args:
            parents (list): lista de padres
            diccionario (dict): diccionario del dataset

        Returns:
            list: individuos tras la mutacion
        """

        for individuo in parents:
            if random.random() < self.mutation_prob:
                if random.random() < 0.5 and len(individuo) < self.numRules:    #add rule
                    rule = self.__generate_rule(diccionario)
                    individuo.append(rule)
                elif len(individuo) > 2:                                        #remove rule
                    individuo.pop(np.random.choice(len(individuo)))
        return parents

    def __mutation(self, parents, diccionario):
        """
            Author:
                Pablo Sanchez

        Args:
            parents (list): lista de los padres
            diccionario (dict): diccionario del dataset

        Returns:
            list: lista de descendientes
        """
        descendents = self.__bitflip_mutation(parents, diccionario)
        descendents = self.__rule_mutation(descendents, diccionario)
        return descendents

    def __elitism(self, fitness_list: list) -> list:
        """
        Author:
            Rafael Dominguez

        Args:
            fitness_list (list): lista de fitnesses de cada uno de los individuos

        Returns:
            list: mejores individuos de la generación
        """

        num_elits = math.ceil(len(fitness_list)*self.elit_prob)
        elite_list = []

        fitness_list_copy = copy.deepcopy(fitness_list)
        population_list_copy = copy.deepcopy(self.population)

        for _ in range(num_elits):
            elit = max(fitness_list_copy)
            elit_index = fitness_list_copy.index(elit)
            fitness_list_copy.remove(elit)
            elite_list.append(population_list_copy.pop(elit_index))


        return elite_list

    def __predict(self, line: np.ndarray, individual: list, diccionario: dict):
        """
        Author:
            Rafael Dominguez

        Args:
            line (np.ndarray): linea del dataset
            individual (list): individuo de la poblacion
            diccionario (dict): diccionario del dataset

        Returns:
            Literal[0,1] | None: 0 si predice False, 1 si predice True y None si no es capaz de predecir esa linea
        """

        predicted_classes = []
        predicted_rules = []
        for rule in individual:
            line_rule = self.__generate_rule(diccionario, line)
            if self.__isTrue(rule, line_rule):
                predicted_rules.append(rule[-1])

        if not predicted_rules:
            return None
        
        counts = np.bincount(predicted_rules)
        predicted_classes.append(np.argmax(counts))

        if sum(predicted_classes) < len(predicted_classes)/2:
            return 0
        else:
            return 1

    def __fitness(self, xdata: np.ndarray , ydata: np.ndarray, diccionario):
        """
        Author:
            Rafael Dominguez

        Calcula el fitness de cada uno de los individuos de la poblacion y crea la lista fitness

        Args:
            xdata (np.ndarray): datos del dataset
            ydata (np.ndarray): columna clase del dataset
            diccionario (dict): diccionario del dataset

        Returns:
            list: lista fitness
        """
        
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

    def __populate(self, diccionario):
        """
        Author:
            Rafael Dominguez Saez

        Genera una poblacion de individuos con sus propias reglas

        Args:
            diccionario (dict): diccionario del dataset
        """        
        self.population = []
        for _ in range(self.numPopulation):
            individial = []

            n_rules = random.randint(1,self.numRules)
            for _ in range(n_rules):
                rule = self.__generate_rule(diccionario)

                individial.append(rule)

            self.population.append(individial)

    def __generate_rule(self, diccionario:dict, line=None):
        """
        Author:
            Rafael Dominguez

        En caso de dar una linea, crea la rule de esa linea, en caso contrario, genera una regla aleatoria

        Args:
            diccionario (dict): diccionario del dataset
            line (list, optional): Linea del dataset. Defaults to None.

        Returns:
            list: regla creada a partir de la linea o aleatoria
        """

        rule = []
        if line is not None:
            for i,key in enumerate(diccionario.keys()):
                lkeys = list(diccionario[key].keys())
                if len(line) == i:
                    break
                
                for j in range(len(lkeys)):
                    if line[i] == j:
                        rule.append(1)
                    else:
                        rule.append(0)
        else:
            for key in diccionario.keys():
                len_lkeys = len(list(diccionario[key].keys()))
                rule_part = [0]*len_lkeys
                rand_index = random.randint(0, len_lkeys-1)
                rule_part[rand_index] = 1
                rule.extend(rule_part)
        return rule



    def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos, diccionario:dict):
        x_train = datosTrain.iloc[:, :-1].values
        y_train = datosTrain.iloc[:, -1].values

        self.__populate(diccionario)
        if self.debug_fitness:
            self.debug_fitness_list = []

        for _ in range(self.epoches):

            fitness_list = self.__fitness(x_train, y_train, diccionario)

            elit_list = self.__elitism(fitness_list)

            if self.debug_fitness:
                self.debug_fitness_list.append((max(fitness_list), sum(fitness_list)/len(fitness_list)))

            #Parents_selection: Obtenemos los padres
            parents = self.__parents_selection(fitness_list)
            #Crossover: cruce a partir de los padres crear nuevas soluciones
            parents = self.__crossover(parents)
            #Mutations: los padres reciben mutaciones
            descendants = self.__mutation(parents, diccionario)
            #Survivors: purga de los malardos (union de progenitores, descendientes y elite)
            survivors = descendants + elit_list

            #Los supervivientes pasa a ser una nueva poblacion
            self.population = survivors

        fitness_list = self.__fitness(x_train, y_train, diccionario)

        if self.debug_fitness:
                self.debug_fitness_list.append((max(fitness_list), sum(fitness_list)/len(fitness_list)))

        if self.math_related_prediction:
            elit = max(fitness_list)
            elit_index = fitness_list.index(elit)

            self.theOnes = [self.population[elit_index]]
            if self.debug_rules:
                print("-Elegido para clasificar->",self.theOnes[0], \
                      "\n-Porcentaje de aciertos ->", elit)
        else:
            self.theOnes = self.__elitism(fitness_list)
            if self.debug_rules:
                print("-Elegidos para clasificar->",self.theOnes, \
                      "\n-Porcentaje de aciertos->", sorted(fitness_list,reverse=True)[:math.ceil(self.numPopulation*self.elit_prob)])

    def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
        x_test = datosTest.iloc[:, :-1].values

        pred = []
        predicted = []
        for data in x_test:
            if self.math_related_prediction:
                pred.append(self.__predict(data, self.theOnes[0], diccionario))
            else:
                for elit in self.theOnes:
                    result = self.__predict(data, elit, diccionario)
                    if result is not None:
                        predicted.append(result)
                if sum(predicted) >= len(predicted)/2:
                    pred.append(1)
                else:
                    pred.append(0)

        return np.asarray(pred, dtype="object")

