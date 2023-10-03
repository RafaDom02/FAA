from abc import ABCMeta,abstractmethod
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
    pass
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):
       
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
    pass  

class ClasificadorNaiveBayes (Clasificador):
  def __init__(self, LaPlace: bool):
    self.LaPlace = LaPlace
  def entrenamiento(self,datosTrain: pd.DataFrame,nominalAtributos,diccionario):
    probsClase = {}
    # datosTrain (dataframe) get average of the last column of the dataframe
    priori_class = datosTrain.iloc[:, -1].value_counts(normalize=True)
    condicionales = []
    print(priori_class)

    print(datosTrain.shape[0])
    print(datosTrain.shape[1])

    for i in range(datosTrain.shape[1]-1):
      # obtener las probabilidades condicionales de datosTrain
      if nominalAtributos[i] == True:
        tabla = np.zeros(len(diccionario[i]), len(diccionario[i])-1)
        for row in datosTrain:
          fila = int(row[i])
          columna = int(row[-1])
          tabla[fila][columna] += 1
      else:
        pass
    

  def clasifica(self,datosTest,nominalAtributos,diccionario):
    pass
  


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