import copy
import numpy as np
import random
from Clasificador import Clasificador
import math
from typing import Any
import pandas as pd

class ClasificadorGenetico(Clasificador):
    def __init__(self, numPopulation: int = 50,  epoches: int = 50, numRules: int = 5, \
                elit_prob: float = 0.05, cross_prob: float = 0.02, mutation_prob: float = 0.05, \
                bitmut_prob: float = 0.15, math_related_prediction: bool = True) -> Any:
        self.numPopulation = numPopulation
        self.epoches = epoches
        self.numRules = numRules
        self.elit_prob = elit_prob
        self.cross_prob = cross_prob
        self.mutation_prob = mutation_prob
        self.bitmut_prob = bitmut_prob
        self.math_related_prediction = math_related_prediction

    def __isTrue(self, ruleIndividual, ruleLine) -> bool:
        for i in range(len(ruleIndividual)-2):
            if ruleIndividual[i] != 0:
                if ruleIndividual[i] == ruleLine[i]:
                    return False
        return True

    def __parents_selection(self, fitness_list):

        # de la lista fitness, hacemos la suma de sus elementos 
        # y calculamos el peso de cada uno de los elementos
        # -> ((fitness elemento)/(suma fitness elementos))

        # con np.random.choice obtenemos de la lista self.population
        # los padres teniendo en cuenta cada peso calculado antes

        # devolvemos los padres

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

        descendents = []
        # Para cada uno de los padres, se seleccionan 2
            # tiramos un np.random.choice para ver si se hace el crossover
            # (teniendo en cuenta self.cross_prob)

            # si sale que se hace crossover ya decides tu si se hace inter o intra
            # y ponemos los descendientes en la lista de descendientes (mucha suerte con esto)

            # en caso de no hacerse crossover se devuelven los padres
        
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
    
    def __bitflip_mutation(self, parents):

        # para cada individuo de los padres
            # por cada una de las reglas de los padres
                # por cada bit de una regla
                    # se hace np.random.choice para ver si se hace flip del bit
                    # con self.bitflip_prob
                    # si se hace flip se hace el flip del bit
        for individuo in parents:
            for regla in individuo:
                for bit in regla:
                    if random.random() < self.bitmut_prob:
                        bit = 1 - bit
        return parents


    def __rule_mutation(self, parents, diccionario):

        # mah o menoh parecio al bitflip_mutation pero mas chungo
        # para cada individuo de los padres
            # se hace np.random.choice para ver si se añade o se elimina una regla
            # con self.mutation_prob
            # si se añade una regla se añade una regla con np.random.choice
            # si se elimina una regla se elimina una regla con np.random.choice
        for individuo in parents:
            if random.random() < self.mutation_prob:
                if random.random() < 0.5 and len(individuo) < self.numRules: #add rule
                    rule = []
                    # while sum(rule) == 0 or sum(rule) == len(rule):
                    while sum(rule) == 0:
                        rule = self.__generate_rule(diccionario)
                    individuo.append(rule)
                elif len(individuo) > 2: #remove rule
                    individuo.pop(np.random.choice(len(individuo)))
                    if not isinstance(individuo[0], list):
                        individuo = [individuo] #Arreglamos si desace la lista de 1 cadena
        return parents

    def __mutation(self, parents, diccionario):
        descendents = self.__bitflip_mutation(parents)
        descendents = self.__rule_mutation(descendents, diccionario)
        return descendents

    def __elitism(self, fitness_list: list) -> list:

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

        #print(f"{sum(predicted_classes)} ? {len(predicted_classes)}")

        if sum(predicted_classes) > len(predicted_classes)/2:
            #print(f"{sum(predicted_classes)} > {len(predicted_classes)}")
            return 0
        else:
            #print(f"{sum(predicted_classes)} < {len(predicted_classes)}")
            return 1

    def __fitness(self, xdata: np.ndarray , ydata: np.ndarray, diccionario):
        
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
            Genera una poblacion de individuos con sus propias reglas
        """        
        self.population = []
        for _ in range(self.numPopulation):
            individial = []

            n_rules = random.randint(1,self.numRules)
            for _ in range(n_rules):
                rule = []

                while sum(rule) == 0:
                    rule = self.__generate_rule(diccionario, None)

                individial.append(rule)

            self.population.append(individial)

    def __generate_rule(self, diccionario:dict, line=None):
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
                lkeys = len(list(diccionario[key].keys()))
                rule_part = [0]*lkeys
                rand_index = random.randint(0, lkeys-1)
                rule_part[rand_index] = 1
                rule.extend(rule_part)
        return rule



    def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos, diccionario:dict):
        x_train = datosTrain.iloc[:, :-1].values
        y_train = datosTrain.iloc[:, -1].values

        self.__populate(diccionario)

        for _ in range(self.epoches):

            fitness_list = self.__fitness(x_train, y_train, diccionario)

            elit_list = self.__elitism(fitness_list)

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

        if self.math_related_prediction:
            elit = max(fitness_list)
            elit_index = fitness_list.index(elit)

            self.theOnes = [self.population[elit_index]]
            print(self.theOnes, elit)
        else:
            #TODO
            pass

    def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
        x_test = datosTest.iloc[:, :-1].values

        pred = []
        predicted = []
        for data in x_test:
            if self.math_related_prediction:
                pred.append(self.__predict(data, self.theOnes[0], diccionario))
            else:
                #TODO
                pass

        return np.asarray(pred, dtype="object")

