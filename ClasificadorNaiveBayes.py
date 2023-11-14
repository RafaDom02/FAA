from scipy.stats import norm
import numpy as np
from Clasificador import Clasificador
import math
import pandas as pd

class ClasificadorNaiveBayes (Clasificador):
  """Clasificador de propia implementación segun el algoritmo de Naive Bayes
  Authors: Rafael Dominguez

  Args:
      Clasificador (_type_): Clase padre de clasificador

  """
  def __init__(self, LaPlace: bool = False):
    self.LaPlace = LaPlace  # Guarda si se usa La Place en el entrenamiento
    self.prioris = None     # Guarda los prioris de los datos
    self.likelihoods = None # Guarda los likelihoods obtenidos en el entrenamiento

  def _multinomialNB(self, x_dat: pd.DataFrame, y_dat: pd.DataFrame, idx: int, diccionario: dict):
    """
    Parte del entrenamiento que se encarga de los datos multinomiales para Naive Bayes
    Authors: Rafael Dominguez
    Args:
        x_dat (pd.DataFrame): Con los datos sin la columna Clase
        y_dat (pd.DataFrame): Con la columna Clase
        idx (int): indice
        diccionario (dict): diccionario de la clase Datos

    Returns:
        NDArray: Contiene la tabla con los valores obtenidos, se aplica La Place si su flag esta a True y hay 1 valor 0 
    """        
    n_xidx = len(diccionario[list(diccionario.keys())[idx]])
    n_classes = len(diccionario['Class'])
    tabla = np.zeros((n_xidx, n_classes))
    for value in diccionario[list(diccionario.keys())[idx]]:
        val_idx = diccionario[list(diccionario.keys())[idx]][value]
        for class_name in diccionario['Class']:
            class_idx = diccionario['Class'][class_name]
            tabla[val_idx, class_idx] = sum((x_dat.iloc[:,idx] == val_idx)&(y_dat == class_idx))/sum(y_dat == class_idx)

    if self.LaPlace and np.any(tabla == 0):
        print("SE EJECUTA LA PLACE!!!")
        print(tabla, "\n")
        tabla += np.ones((n_xidx, n_classes))
        print(tabla, "\n")

    return tabla

  def _gaussianNB(self, x_dat: pd.DataFrame, y_dat: pd.DataFrame, idx: int, diccionario: dict):
    """
    Parte del entrenamiento que se encarga de los datos gaussianos para Naive Bayes
    Authors: Pablo Sanchez
    Args:
        x_dat (pd.DataFrame): Con los datos sin la columna clases
        y_dat (pd.DataFrame): Con la columna clases
        idx (int): indice
        diccionario (dimport mathict): diccionario de la clase Datos

    Returns:
        NDArray: Contiene la tabla con los valores obtenidos guardando la media y la varianza
    """    
    n_classes = len(diccionario['Class'])

    tabla = np.zeros((n_classes, 2)) # 2 columns: mean and variance for each class

    for class_name in diccionario['Class']:
        class_idx = diccionario['Class'][class_name]
        mean_sum = sum(elem for (idx, elem) in enumerate(x_dat.iloc[:,idx]) if y_dat.iloc[idx]==class_idx)
        mean_cl = mean_sum/sum(y_dat == class_idx)
        variance_sum = sum((elem-mean_cl)**2 for (idx, elem) in enumerate(x_dat.iloc[:,idx]) if y_dat.iloc[idx]==class_idx)
        variance_cl = variance_sum/sum(y_dat == class_idx)

        tabla[class_idx][0] = mean_cl
        tabla[class_idx][1] = variance_cl

    return tabla

  def entrenamiento(self,datosTrain: pd.DataFrame,nominalAtributos,diccionario):
    """
    Entrena para un dataset
    Authors: Pablo Sanchez
    Args:
        datosTrain (pd.DataFrame): Datos para el entrenamiento
        nominalAtributos (list): Contiene si un atributo en su columna es nominal o no
        diccionario (dict): diccionario de la clase Datos
    """    
    data_wo_lastColumn = datosTrain.iloc[:,:-1] #Data train sin la columna de las clases :(
    data_lastColumn = datosTrain.iloc[:,-1]     #Columna de las classes :)
    aux = []
    self.prioris = datosTrain.iloc[:, -1].value_counts(normalize=True) #Los prioris
    
    for idx in range(data_wo_lastColumn.shape[1]):
            if nominalAtributos[idx]:
                tabla = self._multinomialNB(data_wo_lastColumn, data_lastColumn, idx, diccionario)
            else:
                tabla = self._gaussianNB(data_wo_lastColumn, data_lastColumn,idx, diccionario)
            aux.append(tabla)

    self.likelihoods = np.asarray(aux, dtype="object")

  def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
    """
    Determina un dataset dado gracias a los likelihood anteriormente calculados
    Authors: Pablo Sanchez
    Args:
        datosTest (pd.DataFrame): datos para la clasificacion
        nominalAtributos (list): Contiene si un atributo en su columna es nominal o no
        diccionario (dict): diccionario de la clase Datos

    Returns:
        NDArray: Array que contiene los resultados
    """    
    data_wo_lastColumn = datosTest.iloc[:,:-1]

    xdata, ydata = data_wo_lastColumn.shape
    n_classes = len(diccionario['Class'])

    pred = []
    for i in range(xdata):
        classes_probs = []
        for j in range(n_classes):
            class_p = self.prioris[j]
            for idx in range(ydata):
                if nominalAtributos[idx]:
                    class_p *= self.likelihoods[idx][int(data_wo_lastColumn.iloc[i, idx])][j]
                else:
                    mean = self.likelihoods[idx][j][0]
                    var = self.likelihoods[idx][j][1]
                    class_p *= norm.pdf(data_wo_lastColumn.iloc[i, idx], loc=mean, scale=math.sqrt(var))
            classes_probs.append(class_p)
        pred.append(classes_probs.index(max(classes_probs)))

    return np.asarray(pred, dtype="object")
