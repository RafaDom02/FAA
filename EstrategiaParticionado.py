from abc import ABCMeta,abstractmethod
import random
import numpy as np

class Particion():

  # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones  
  def __init__(self):
    self.indicesTrain=None
    self.indicesTest=None

#####################################################################################################

class EstrategiaParticionado:
  
  # Clase abstracta
  __metaclass__ = ABCMeta

  def __init__(self) -> None:
    self.particiones = []
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor 
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):

  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
  def __init__(self, numeroEjecuciones: int, proporcionTest: float) -> None:
    super().__init__()
    self.numeroEjecuciones = numeroEjecuciones
    self.proporcionTest = proporcionTest
    

  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):
    for _ in range(0,self.numeroEjecuciones):
      random.seed(seed)
      part = Particion()
      test = datos.sample(frac = self.proporcionTest)
      train = datos.drop(test.index)

      part.indicesTest = test.index
      part.indicesTrain = train.index
      self.particiones.append(part)
            
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):

  # Crea particiones segun el metodo de validacion cruzada.
  def __init__(self, numeroParticiones: int) -> None:
    super().__init__()
    self.numeroParticiones = numeroParticiones

  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):   
    part_length = len(datos) // self.numeroParticiones

    data_sublists = [datos[i:i + part_length] for i in range(0, len(datos), part_length)]
 
    for i in range(0,self.numeroParticiones):
      random.seed(seed)
      part = Particion()
      part.indicesTest = data_sublists[i].index
      part.indicesTrain = datos.drop(data_sublists[i].index).index
      self.particiones.append(part)


    
