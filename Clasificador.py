from abc import ABCMeta,abstractmethod
import math
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

from sklearn.naive_bayes import MultinomialNB, GaussianNB, CategoricalNB

from Datos import Datos

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
    particionado.creaParticiones(dataset.datos,seed)
    errores = []
    print("Numero de particiones realizadas:", len(particionado.particiones))
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
    self.LaPlace = LaPlace  # Guarda si se usa La Place en el entrenamiento
    self.prioris = None     # Guarda los prioris de los datos
    self.likelihoods = None # Guarda los likelihoods obtenidos en el entrenamiento

  def _multinomialNB(self, x_dat: pd.DataFrame, y_dat: pd.DataFrame, idx: int, diccionario: dict):
    """
    Parte del entrenamiento que se encarga de los datos multinomiales para Naive Bayes

    Args:
        x_dat (pd.DataFrame): Con los datos sin la columna clases
        y_dat (pd.DataFrame): Con la columna clases
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

    Args:
        x_dat (pd.DataFrame): Con los datos sin la columna clases
        y_dat (pd.DataFrame): Con la columna clases
        idx (int): indice
        diccionario (dict): diccionario de la clase Datos

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

class ClasificadorNaiveBayesSKLearn(Clasificador):
  def __init__(self, clasificador, LaPlace=True, prior=True):
    if clasificador == 1:
      self.clasificador = MultinomialNB(alpha=int(LaPlace), fit_prior = prior)
    elif clasificador == 2:
      self.clasificador = GaussianNB()
    else:
      self.clasificador = CategoricalNB(alpha=int(LaPlace), fit_prior = prior)

  def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos: list, diccionario: dict):
    data_wo_lastColumn = datosTrain.iloc[:,:-1] #Data train sin la columna de las clases :(
    data_lastColumn = datosTrain.iloc[:,-1]     #Columna de las classes :)

    self.clasificador.fit(data_wo_lastColumn, data_lastColumn)

  def clasifica(self,datosTest: pd.DataFrame, nominalAtributos: list, diccionario: dict):
    data_wo_lastColumn = datosTest.iloc[:,:-1] #Data train sin la columna de las clases :(
    return self.clasificador.predict(data_wo_lastColumn)

class ClasificadorKNN (Clasificador):
  def entrenamiento(self,datosTrain,nominalAtributos,diccionario):
    pass

  def clasifica(self,datosTest,nominalAtributos,diccionario):
    pass
  