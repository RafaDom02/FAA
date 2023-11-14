from typing import Any
import numpy as np
from Clasificador import Clasificador
import pandas as pd


class ClasificadorKNN (Clasificador):
  """Clasificador basado en KNN
  Authors: Rafael Dominguez
  """
  def __init__(self, k = 3) -> Any:
    self.k = k

  def _euclidean_distance(self, line_test: np.ndarray, row_train: int):
    """
    Calcula la distancia euclidiana entre dos puntos
    Authors: Rafael Dominguez
    Args:
        line_test (np.ndarray): Punto de test
        row_train (int): Punto de train
    Returns:
        float: Distancia euclidiana entre los dos puntos
    """
    
    data_train = self.train_wo_lastColumn.values  # Obtenemos los valores del DataFrame como una matriz NumPy
    line_train = data_train[row_train, :]  # Obtenemos la primera línea como un array

    # Calculamos la distancia euclidiana utilizando la función de numpy
    distance = np.linalg.norm(line_test - line_train)

    return distance

  def entrenamiento(self,datosTrain: pd.DataFrame,nominalAtributos,diccionario):
    self.train_wo_lastColumn = datosTrain.iloc[:,:-1] #Data train sin la columna de las clases :(
    self.train_lastColumn = datosTrain.iloc[:,-1]     #Columna de las classes :)
    return

  def clasifica(self,datosTest: pd.DataFrame,nominalAtributos,diccionario):
    """
    Clasifica los datos dados usando KNN

    Author:
        Rafael Dominguez Saez
    
    Args:
        datosTest (pd.DataFrame): Datos ha clasificar
        nominalAtributos (_type_): No usado
        diccionario (_type_): No usado

    """    
    data_wo_lastColumn = datosTest.iloc[:,:-1] #Data train sin la columna de las clases :(
    pred = []

    line_test = data_wo_lastColumn.shape[0]
    line_train = self.train_wo_lastColumn.shape[0]
    for idx_test in range(line_test):
      distances = []
      for idx_train in range(line_train):
        distances.append(self._euclidean_distance(data_wo_lastColumn.values[idx_test], idx_train))
      indices_of_smallest = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k] #Obtenemos el indice de los datos mas proximos por distancia euclidea
      neighbors_aux = []
      for i in indices_of_smallest:
        neighbors_aux.append(self.train_lastColumn.iloc[i])
      # Obtener clase mas comun
      most_common = max(set(neighbors_aux), key= neighbors_aux.count)
      pred.append(most_common)

    return np.asarray(pred, dtype="object")