from abc import ABCMeta,abstractmethod
import random
import numpy as np
import pandas as pd

from sklearn.naive_bayes import MultinomialNB, GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


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
    for part in particionado.particiones:
      datTrain = dataset.extraeDatos(part.indicesTrain)
      datTest = dataset.extraeDatos(part.indicesTest)
      
      clasificador.entrenamiento(datTrain,dataset.nominalAtributos, dataset.diccionarios)
      pred = clasificador.clasifica(datTest,dataset.nominalAtributos, dataset.diccionarios)
      error = clasificador.error(datTest.iloc[:,-1], pred)
      errores.append(error)
    
    return errores

class ClasificadorNaiveBayesSKLearn(Clasificador):
  """Clasificador basado en Naive Bayes usando las implementaciones de la librería Sciki-learn
  Authors: Rafael Dominguez

  Args:
      Clasificador (_type_): _description_
  """
  def __init__(self, clasificador, LaPlace=True, prior=True):
    if clasificador == 1:
      self.clasificador = MultinomialNB(alpha=int(LaPlace), fit_prior = prior)
    elif clasificador == 2:
      self.clasificador = GaussianNB()
    else:
      self.clasificador = CategoricalNB(alpha=int(LaPlace), fit_prior = prior)

  def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos: list, diccionario: dict):
    """Entrenamiento para un dataset
    Authors: Rafael Dominguez
    Args:
        datosTrain (pd.DataFrame): Datos para el entrenamiento
        nominalAtributos (list): Contiene si un atributo en su columna es nominal o no
        diccionario (dict): diccionario de la clase Datos
    """
    data_wo_lastColumn = datosTrain.iloc[:,:-1] #Data train sin la columna de las clases :(
    data_lastColumn = datosTrain.iloc[:,-1]     #Columna de las classes :)

    self.clasificador.fit(data_wo_lastColumn, data_lastColumn)

  def clasifica(self,datosTest: pd.DataFrame, nominalAtributos: list, diccionario: dict):
    """Determina un dataset dado gracias a los likelihood anteriormente calculados
    Authors: Rafael Dominguez
    Args:
        datosTest (pd.DataFrame): datos para la clasificacion
        nominalAtributos (list): Contiene si un atributo en su columna es nominal o no
        diccionario (dict): diccionario de la clase Datos
    Returns:
        NDArray: Array que contiene los resultados
    """
    data_wo_lastColumn = datosTest.iloc[:,:-1] #Data train sin la columna de las clases :(
    return self.clasificador.predict(data_wo_lastColumn)
  

class ClasificadorKNNSKLearn(Clasificador):
  """
  Author:
      Pablo Sánchez Fernández del Pozo
  """
  def __init__(self, k=3):
      self.k = k
      self.model = KNeighborsClassifier(n_neighbors=k)

  def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos, diccionario):
      x = datosTrain.iloc[:, :-1]
      y = datosTrain.iloc[:, -1]

      self.model.fit(x, y)

  def clasifica(self, datosTest: pd.DataFrame, nominalAtributos, diccionario):
      X_test = datosTest.iloc[:, :-1]
      y_pred = self.model.predict(X_test)

      return np.asarray(y_pred, dtype="object")
  
class ClasificadorRegresionLogisticaSKLearn(Clasificador):
  def __init__(self, epocas:int = 100):
    self.model = LogisticRegression(C=0.5, max_iter=epocas)


  def entrenamiento(self, datosTrain: pd.DataFrame, nominalAtributos, diccionario):
    x = datosTrain.iloc[:, :-1]
    y = datosTrain.iloc[:, -1]
    
    self.model.fit(x,y)

  def clasifica(self, datosTest: pd.DataFrame, nominalAtributos, diccionario):
    x_test = datosTest.iloc[:, :-1]
    y_pred = self.model.predict(x_test)

    return np.asarray(y_pred, dtype="object")