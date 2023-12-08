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
                bitmut_prob: float = 0.15) -> Any:
        self.numPopulation = numPopulation
        self.epoches = epoches
        self.numRules = numRules
        self.elit_prob = elit_prob
        self.cross_prob = cross_prob
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

        # de la lista fitness, hacemos la suma de sus elementos 
        # y calculamos el peso de cada uno de los elementos
        # -> ((fitness elemento)/(suma fitness elementos))

        # con np.random.choice obtenemos de la lista self.individuals
        # los padres teniendo en cuenta cada peso calculado antes

        # devolvemos los padres

        # Calcula la suma total de los valores de fitness
        total_fitness = sum(fitness_list)

        # Calcula los pesos relativos de cada individuo
        weights = [fitness / total_fitness for fitness in fitness_list]

        # Selecciona dos padres usando np.random.choice con los pesos calculados
        selected_parents_indices = np.random.choice(len(self.individuals), size=2, p=weights)

        # Obtiene los individuos correspondientes a los índices seleccionados
        parent1 = self.individuals[selected_parents_indices[0]]
        parent2 = self.individuals[selected_parents_indices[1]]

        return parent1, parent2

    def __crossover(self, parents):
        #TODO: hace el crossover

        descendents = []
        # Para cada uno de los padres, se seleccionan 2
            # tiramos un np.random.choice para ver si se hace el crossover
            # (teniendo en cuenta self.cross_prob)

            # si sale que se hace crossover ya decides tu si se hace inter o intra
            # y ponemos los descendientes en la lista de descendientes (mucha suerte con esto)

            # en caso de no hacerse crossover se devuelven los padres
        
        # Pair in tuples all parents in a random way
        PairsOfParents = np.random.choice(parents, size=(len(parents)//2, 2), replace=False)
        for padre1, padre2 in PairsOfParents:
            #get random number from 0 to 1 and check if its less than cross_prob
            if random.random() < self.cross_prob:
                #get random number from 0 to numRules
                cross_point = random.randint(0, self.numRules)
                #intra crossover
                if random.random() < 0.5:
                    descendents.append(padre1[:cross_point] + padre2[cross_point:])
                    descendents.append(padre2[:cross_point] + padre1[cross_point:])
                #inter crossover
                else:
                    descendents.append(padre1[:cross_point] + padre2[cross_point:])
                    descendents.append(padre2[:cross_point] + padre1[cross_point:])
            else:
                descendents.append(padre1)
                descendents.append(padre2)

        return descendents
    
    def __bitflip_mutation(self, parents):
        #TODO: mutacion de reglas con el bitflip

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


    def __rule_mutation(self, parents):
        #TODO: añade o elimina una regla a los padres

        # mah o menoh parecio al bitflip_mutation pero mas chungo
        # para cada individuo de los padres
            # se hace np.random.choice para ver si se añade o se elimina una regla
            # con self.mutation_prob
            # si se añade una regla se añade una regla con np.random.choice
            # si se elimina una regla se elimina una regla con np.random.choice
        for individuo in parents:
            if random.random() < self.mutation_prob:
                if random.random() < 0.5: #add rule
                    individuo.append(np.random.choice([0,1], k=self.rules_length))
                else: #remove rule
                    individuo.pop(np.random.choice(len(individuo)))
        return parents

    def __mutation(self, parents):
        descendents = self.__bitflip_mutation(parents)
        descendents = self.__rule_mutation(descendents)
        return descendents

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
        #TODO: REVISAR y ARREGLAR, solo funciona con datos en binario, esto lo hago yo
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
            
            #TODO: Parents_selection: Obtenemos los padres
            parents = self.__parents_selection(fitness_list)

            #TODO: Crossover: cruce a partir de los padres crear nuevas soluciones
            parents = self.__crossover(parents)

            #TODO: Mutations: los padres reciben mutaciones
            descendants = self.__mutation(parents)

            #TODO: Survivors: purga de los malardos (union de progenitores y elite)
            survivors = descendants

            #TODO: Los supervivientes pasa a ser una nueva poblacion
            self.population = survivors

    def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
        pass



if __name__ == "__main__":
    generic = ClasificadorGenetico()
    print(generic.__parents_selection())
