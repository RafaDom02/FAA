from abc import ABCMeta,abstractmethod
import math
import random
import numpy as np
import pandas as pd

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
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
    # devuelve el error
    return 1-sum(datos == pred)/len(pred)
    
    
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
    
    for part in particionado.particiones:
      datTrain = dataset.extraeDatos(part.indicesTrain)
      datTest = dataset.extraeDatos(part.indicesTest)
      
      clasificador.entrenamiento(datTrain,dataset.nominalAtributos, dataset.diccionarios)
      pred = clasificador.clasifica(datTest,dataset.nominalAtributos, dataset.diccionarios)
      
      error = clasificador.error(datTest[:-1], pred)
      errores.append(error)
    
    return errores


class ClasificadorNaiveBayes (Clasificador):
  def __init__(self, LaPlace: bool = False):
    self.LaPlace = LaPlace
    self.prioris = None
    self.trainData = []
 
  def entrenamiento(self,datosTrain: pd.DataFrame,nominalAtributos,diccionario):
    data_wo_lastColumn = datosTrain[:,:-1] #Data train sin la columna de las clases :(
    data_lastColumn = datosTrain[:,-1] #Columna de las classes :)
    n_classes = len(diccionario[-1]) #Numero de clases con las que tratamos 

    self.prioris = datosTrain.iloc[:, -1].value_counts(normalize=True) #Los prioris
    """for key in diccionario[-1].keys():
      aux = diccionario[-1][key]
      self.prioris = aux"""
    
    for i in range(data_wo_lastColumn.shape[1]):
      if nominalAtributos[i] == True:
        tabla = np.zeros((len(diccionario[i]), len(diccionario[-1])))
        for value in diccionario[i]:
          val_id = diccionario[i][value]
          for cl in diccionario[-1]:
            cl_id = diccionario[-1][cl]
            tabla[val_id][cl_id] = sum((data_wo_lastColumn[:,i] == val_id)&(data_lastColumn[:,i] == cl_id)) \
            											/sum(data_lastColumn == cl_id)
        if self.LaPlace and np.any(tabla == 0):
          tabla+=1
      else:
        tabla = np.zeros((n_classes, 2))
        for cl in diccionario[-1]:
          cl_id = diccionario[-1][cl]
          
          m_sum = sum(elem for (id, elem) in enumerate(data_wo_lastColumn[:,i]) if data_lastColumn[id]==cl_id)
          m_cl = m_sum/sum(data_lastColumn == cl_id)
          
          var_sum = sum((elem-m_cl)**2 for (id, elem) in enumerate(data_wo_lastColumn[:,i]) if data_lastColumn[id]==cl_id)
          var_cl = var_sum/sum(data_lastColumn == cl_id)
          
          tabla[cl_id][0] = m_cl
          tabla[cl_id][1] = var_cl

          self.trainData.append(tabla)
          
    self.trainData = np.asarray(self.trainData, dtype="object")
          
            

  def clasifica(self,datosTest,nominalAtributos,diccionario):
    predicciones = []
    for datos in datosTest:
      prioriDict = {}
      for j,key in enumerate(diccionario[-1].keys()):
        val = diccionario[-1][key]
        p = 1
        for i in range(len(datos)-1):
          if nominalAtributos[i] == True:
            ver = self.trainData[i][int(datos[i]), val]
            evi = sum(self.trainData[i][:, val])
            p*=(ver / evi)
          else:
            #
            sqrt = math.sqrt(2*math.pi*self.trainData[i][1,val])
            #e^(-(valor-media)^2)/2*desviacion)
            exp = math.exp(-1*pow(datos[i]-self.trainData[i][0,val], 2))/(2*self.trainData[i][1,val])
            p*=(sqrt/exp)
        
        p*=self.prioris[j]
        prioriDict[key]=p
      max_class = max(prioriDict, key=lambda k: prioriDict[k])
      predicciones.append(max_class)
    return np.array(predicciones)



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