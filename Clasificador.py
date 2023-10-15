from abc import ABCMeta,abstractmethod
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

from Datos import Datos
import EstrategiaParticionado

class Clasificador:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto. Crea el modelo a partir de los datos de entrenamiento
  # datosTrain: matriz numpy o dataframe con los datos de entrenamiento
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datosTrain,nominalAtributos,diccionario):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto. Devuelve un numpy array con las predicciones
  # datosTest: matriz numpy o dataframe con los datos de validaci�n
  # nominalAtributos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  # devuelve un numpy array o vector con las predicciones (clase estimada para cada fila de test)
  def clasifica(self,datosTest,nominalAtributos,diccionario):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self,datos: pd.DataFrame,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
    # devuelve el error
    return 1-(datos == pred).sum() /len(pred)
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset: Datos,clasificador,seed=None):
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opci�n es repetir la validaci�n simple un n�mero especificado de veces, obteniendo en cada una un error. Finalmente se calcular�a la media.
    # devuelve el vector con los errores por cada partici�n
    
    # pasos
    # crear particiones
    # inicializar vector de errores
    # for cada partici�n
    #     obtener datos de train
    #     obtener datos de test
    #     entrenar sobre los datos de train
    #     obtener prediciones de los datos de test (llamando a clasifica)
    #     a�adir error de la partici�n al vector de errores
    random.seed(seed)
    #np.random.shuffle(dataset.datos)
    particionado.creaParticiones(dataset.datos,seed)
    errores = []
    print("Numero de particiones:", len(particionado.particiones))
    for part in particionado.particiones:
      datTrain = dataset.extraeDatos(part.indicesTrain)
      datTest = dataset.extraeDatos(part.indicesTest)
      
      clasificador.entrenamiento(datTrain,dataset.nominalAtributos, dataset.diccionarios)
      pred = clasificador.clasifica(datTest,dataset.nominalAtributos, dataset.diccionarios)
      error = clasificador.error(datTest.iloc[:,-1], pred)
      errores.append(error)
    
    return errores


class ClasificadorNaiveBayes (Clasificador):
  def __init__(self, LaPlace: bool = False):
    self.LaPlace = LaPlace
    self.prioris = None
    self.trainData = []
    self.likelihoods = None

  def _multinomialNB(self, x_dat: pd.DataFrame, y_dat: pd.DataFrame, idx: int, diccionario: dict):
        n_xi = len(diccionario[list(diccionario.keys())[idx]])
        n_classes = len(diccionario['Class'])
        tabla = np.zeros((n_xi, n_classes))
        for value in diccionario[list(diccionario.keys())[idx]]:
            val_idx = diccionario[list(diccionario.keys())[idx]][value]
            for class_name in diccionario['Class']:
                class_idx = diccionario['Class'][class_name]
                tabla[val_idx, class_idx] = sum((x_dat.iloc[:,idx] == val_idx)&(y_dat == class_idx))/sum(y_dat == class_idx)

        if self.LaPlace and np.any(tabla == 0):
            tabla += np.ones((n_xi, n_classes))

        return tabla

  def _gaussianNB(self, x_dat, y_dat, idx, diccionario):
        n_classes = len(diccionario['Class'])

        tabla = np.zeros((n_classes, 2)) # 2 columns: mean and variance for each class

        for class_name in diccionario['Class']:
            class_idx = diccionario['Class'][class_name]
            mean_sum = sum(elem for (idx, elem) in enumerate(x_dat[:,idx]) if y_dat[idx]==class_idx)
            mean_cl = mean_sum/sum(y_dat == class_idx)
            variance_sum = sum((elem-mean_cl)**2 for (idx, elem) in enumerate(x_dat[:,idx]) if y_dat[idx]==class_idx)
            variance_cl = variance_sum/sum(y_dat == class_idx)

            tabla[class_idx][0] = mean_cl
            tabla[class_idx][1] = variance_cl

        return tabla

  def entrenamiento(self,datosTrain: pd.DataFrame,nominalAtributos,diccionario):
    data_wo_lastColumn = datosTrain.iloc[:,:-1] #Data train sin la columna de las clases :(
    data_lastColumn = datosTrain.iloc[:,-1] #Columna de las classes :)
    aux = []
    self.prioris = datosTrain.iloc[:, -1].value_counts(normalize=True) #Los prioris
    """for key in diccionario[-1].keys():
      aux = diccionario[-1][key]
      self.prioris = aux"""
    
    for idx in range(data_wo_lastColumn.shape[1]):
            if nominalAtributos[idx]:
                # calculating frequentist probs for discrete features
                tabla = self._multinomialNB(data_wo_lastColumn, data_lastColumn, idx, diccionario)
            else:
                # calculating means and variances for continuous features
                tabla = self._gaussianNB(data_wo_lastColumn, data_lastColumn,idx, diccionario)

            aux.append(tabla)

    self.likelihoods = np.asarray(aux, dtype="object")
          
            

  def clasifica(self,datosTest: pd.DataFrame,nominalAtributos: list,diccionario: dict):
    data_wo_lastColumn = datosTest.iloc[:,:-1] # all rows, all columns but last one

    ndata, n_feat = data_wo_lastColumn.shape     # number of examples, number of features
    n_classes = len(diccionario['Class'])  # number of different classes

    pred = []
    for i in range(ndata):
        classes_probs = []
        for j in range(n_classes):
            class_p = self.prioris[j]
            for idx in range(n_feat):
                if nominalAtributos[idx]:
                    class_p *= self.likelihoods[idx][int(data_wo_lastColumn.iloc[i, idx])][j]
                else:
                    mean = self.likelihoods[idx][j][0]
                    var = self.likelihoods[idx][j][1]
                    class_p *= norm.pdf(data_wo_lastColumn.iloc[i, idx], loc=mean, scale=math.sqrt(var))
            classes_probs.append(class_p)
        pred.append(classes_probs.index(max(classes_probs)))

    return np.asarray(pred, dtype="object")



class ClasificadorKNN (Clasificador):
  def entrenamiento(self,datosTrain,nominalAtributos,diccionario):
    pass

  def clasifica(self,datosTest,nominalAtributos,diccionario):
    pass
  
if __name__ == "__main__":
  dataset=Datos('./datasets/tic-tac-toe.csv')
  estrategiaVC = EstrategiaParticionado.ValidacionCruzada(10)
  estrategiaVC.creaParticiones(dataset.datos)       
  clasificador = ClasificadorNaiveBayes()
  clasificador.entrenamiento(dataset.datos.loc[estrategiaVC.particiones[0].indicesTrain], dataset.nominalAtributos, dataset.diccionarios)