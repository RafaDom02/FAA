import numpy as np
import random
from Clasificador import Clasificador
import math
from typing import Any
import pandas as pd

class ClasificadorRegresionLogistica(Clasificador):
    def __init__(self, epocas: int = 100, constA: float = 1) -> Any:
        self.epocas = epocas
        self.weights = []
        self.constA = constA

    def _sigmoid(self, x: int):
        if x >= 0:
            return 1 / (1 + math.exp(-x))
        else:
            return 1 / (1 + math.exp(x))
        
    def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos, diccionario):
        
        x_train = datosTrain.iloc[:, :-1]
        y_train = datosTrain.iloc[:, -1]
        
        self.weights = np.array([random.uniform(-0.5, 0.5) for _ in range(x_train.shape[1])])

        for _ in range(self.epocas):
            for j in range(x_train.shape[0]):
                prediction = self._sigmoid(np.dot(x_train.iloc[j], self.weights))
                self.weights -= self.constA * x_train.iloc[j] * (prediction - y_train.iloc[j])
            

    def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
        data_wo_lastColumn = datosTest.iloc[:,:-1]
        pred = []
        for i in range(data_wo_lastColumn.shape[0]):
            sigma = self._sigmoid(np.dot(data_wo_lastColumn.iloc[i], self.weights))
            if sigma >= 0.5:
                result = 1
            else:
                result = 0
            pred.append(result)

        return np.asarray(pred, dtype="object")