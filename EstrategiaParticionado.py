from abc import ABCMeta,abstractmethod
import random
import numpy as np

class Particion():
  """Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones
  Athor: Pablo Sanchez
  """

  def __init__(self):
    """Constructor por omisi�n. Inicializa los vectores con valores nulos.
    Author: Pablo Sanchez
    """
    
    self.indicesTrain=None
    self.indicesTest=None


class EstrategiaParticionado:
  """Esta clase mantiene la lista de particiones del conjunto de particiones
  Athor: Pablo Sanchez
  """
  
  # Clase abstracta
  __metaclass__ = ABCMeta

  def __init__(self) -> None:
    """Constructor por omisi�n. Inicializa la lista de particiones con valores nulos.
    Author: Pablo Sanchez
    """
    self.particiones = []
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor 
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass
  



class ValidacionSimple(EstrategiaParticionado):
  """Implementaci�n de la estrategia de validacion simple (% de test variable).
  Athor: Pablo Sanchez
  """

  def __init__(self, numeroEjecuciones: int, proporcionTest: float) -> None:
    """Constructor de la clase. Inicializa los atributos de la clase.
    Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
    Author: Pablo Sanchez

    Args:
        numeroEjecuciones (int): N�mero de ejecuciones de la estrategia
        proporcionTest (float): Porcentaje de test deseado
    """

    super().__init__()
    self.particiones = []
    self.numeroEjecuciones = numeroEjecuciones
    self.proporcionTest = proporcionTest
    

  def creaParticiones(self,datos,seed=None):
    """Devuelve una lista de particiones (clase Particion)
    Authors: Pablo Sanchez
    Args:
        datos (Array): Datos que particionar
        seed (_type_, optional): _description_. Defaults to None.
    """
    if len(self.particiones) != 0:
      return
    for _ in range(0,self.numeroEjecuciones):
      random.seed(seed)
      part = Particion()
      test = datos.sample(frac = self.proporcionTest)
      train = datos.drop(test.index)

      part.indicesTest = test.index
      part.indicesTrain = train.index
      self.particiones.append(part)
            
      
class ValidacionCruzada(EstrategiaParticionado):
  """Implementaci�n de la estrategia de validacion cruzada.
  Athor: Pablo Sanchez
  """

  # Crea particiones segun el metodo de validacion cruzada.
  def __init__(self, numeroParticiones: int) -> None:
    """Constructor de la clase. Inicializa los atributos de la clase.
    Crea particiones segun el metodo de validacion cruzada.
    Author: Pablo Sanchez
    Args:
        numeroParticiones (int): N�mero de particiones deseadas
    """
    super().__init__()
    self.numeroParticiones = numeroParticiones

  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):
    """Devuelve una lista de particiones (clase Particion)
    Authors: Rafael Dominguez
    Args:
        datos (Array): Datos a particionar
        seed (_type_, optional): _description_. Defaults to None.
    """
    if len(self.particiones) != 0:
      return 
    part_length = len(datos) // self.numeroParticiones

    data_sublists = [datos[i:i + part_length] for i in range(0, len(datos), part_length)]
 
    for i in range(0,self.numeroParticiones):
      random.seed(seed)
      part = Particion()
      part.indicesTest = data_sublists[i].index
      part.indicesTrain = datos.drop(data_sublists[i].index).index
      self.particiones.append(part)


    
