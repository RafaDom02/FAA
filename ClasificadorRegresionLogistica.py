import numpy as np
import random
from Clasificador import Clasificador
import math
from typing import Any
import pandas as pd

class ClasificadorRegresionLogistica(Clasificador):
    def __init__(self, epocas: int = 10, constA: float = 1) -> Any:
        self.epocas = epocas
        self.weights = []
        self.constA = constA

    def _sigmoid(self, x: int):
        z = np.dot(x, self.weights)
        return 1 / (1 + np.exp(-z))
        
    def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos, diccionario):
        
        x_train = datosTrain.iloc[:, :-1].values
        y_train = datosTrain.iloc[:, -1].values
        
        self.weights = np.array([random.uniform(-0.5, 0.5) for _ in range(x_train.shape[1])])

        for _ in range(self.epocas):
            for j in range(x_train.shape[0]):
                prediction = self._sigmoid(x_train[j])
                self.weights -= self.constA * x_train[j] * (prediction - y_train[j])
            

    def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
        data_wo_lastColumn = datosTest.iloc[:,:-1].values
        pred = []
        for i in range(data_wo_lastColumn.shape[0]):
            sigma = self._sigmoid(data_wo_lastColumn[i])

            if sigma >= 0.5:
                result = 1
            else:
                result = 0
            pred.append(result)

        return np.asarray(pred, dtype="object")