import numpy as np
import random
from Clasificador import Clasificador
import math
from typing import Any
import pandas as pd

class ClasificadorGenetico(Clasificador):
    def __init__(self, population: int =50,  epocas: int = 50, numReglas: int = 5, elit_prop: float = 0.05, rule_prob: float = 0.02) -> Any:
        self.population = population
        self.epocas = epocas
        self.numReglas = numReglas
        self.elit_prop = elit_prop
        self.rule_prob = rule_prob

    def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos, diccionario):
        x_train = datosTrain.iloc[:, :-1].values
        y_train = datosTrain.iloc[:, -1].values

        pass

    def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
        pass

