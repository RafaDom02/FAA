import numpy as np
import random
from Clasificador import Clasificador
import math
from typing import Any
import pandas as pd

class ClasificadorRegresionLogistica(Clasificador):
    def __init__(self, n: int, constA: int) -> Any:
        self.n = n
        self.weights = []
        self.constA = constA

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def _productWX(self, X: list) -> int:
        return sum([x*w for x,w in zip(X,self.weights)])
    
    """def _productWX(self, X):
        # Definir la función que calcula el producto punto entre pesos y entrada aquí
        print(X)
        return np.dot(X, self.weights)"""
        
    """def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos, diccionario):
        # Preparación de los datos
        self.weights = np.array([random.uniform(-0.5, 0.5) for _ in range(self.n)])
        X_train = datosTrain.iloc[:, :-1]  # Todas las columnas menos la última.
        y_train = datosTrain.iloc[:, -1]   # La última columna.
        
        # Verificar si el número de pesos coincide con el número de características
        if X_train.shape[1] != len(self.weights):
            len_weights = len(self.weights)
            raise ValueError(f"Incorrect number of elements in weights." +
                            f"Expected {X_train.shape[1]}, got {len_weights}")

        # Proceso de entrenamiento
        for i in range(X_train.shape[0]):
            # Calcula la predicción
            prediction = self._sigmoid(self._productWX(X_train.iloc[i]))
            # Actualiza los pesos para cada característica
            self.weights -= self.constA * X_train.iloc[i] * (prediction - y_train.iloc[i])"""

    def entrenamiento(self,datosTrain: pd.DataFrame,nominalAtributos,diccionario):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(self.n)] # Generamos los pesos iniciales
        data_wo_lastColumn = datosTrain.iloc[:,:-1] #Data train sin la columna de las clases :(
        data_lastColumn = datosTrain.iloc[:,-1]     #Columna de las classes :)

        len_row = data_wo_lastColumn.shape[1]
        len_weights = len(self.weights)
        if len_row != len_weights:
            raise ValueError(f"Incorrect number of elements in weights." +\
                            f"Length of row: {len_row} and length of weights: {len_weights}")
        
        for i in range(data_wo_lastColumn.shape[1]):
            print(datosTrain.iloc[i].tolist())
            for j,w in enumerate(self.weights):
                w -=  self.constA*data_wo_lastColumn.iloc[i].values[j]*\
                (self._sigmoid(
                    self._productWX(data_wo_lastColumn.iloc[i].tolist()))   -
                data_lastColumn.iloc[0])

    def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
        data_wo_lastColumn = datosTest.iloc[:,:-1] #Data train sin la columna de las clases :(
        pred = []
        for i in range(data_wo_lastColumn.shape[0]):
            sigma = self._sigmoid(self._productWX(datosTest.iloc[i].tolist()))
            if sigma >= 0.5:
                result = 1
            else:
                result = 0
            pred.append(result)

        return np.asarray(pred, dtype="object")