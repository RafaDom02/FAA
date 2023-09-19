# -*- coding: utf-8 -*-

# coding: utf-8
import pandas as pd
import numpy as np

import os

class Datos:

    dict = {}
    nominalAtributos = []

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):  
        # Carga el fichero de datos
        self.datos = pd.read_csv(nombreFichero)

        #get a dictionary of labels from the csv file
        aux_list = self.datos.to_dict(orient="records")

        keys = list(aux_list[0].keys())
        #añadir las keys al diccionario
        for i in range(len(keys)):
            self.dict[keys[i]] = {}
        
        for i in range(len(aux_list)):
            for label in self.dict.keys():
                if aux_list[i][label] not in self.dict[label]:
                    if  not (type(aux_list[i][label]) == int or type(aux_list[i][label]) == float) or\
                        label.casefold() == "Class".casefold():

                        if i == 0:
                            self.nominalAtributos.append(True)
                        if aux_list[i][label] not in self.dict[label].values():
                            self.dict[label].update({str(aux_list[i][label]): None})
                    elif (type(aux_list[i][label]) == int or type(aux_list[i][label])) and i == 0:
                            self.nominalAtributos.append(False)
                    else:
                        raise ValueError(f"Error en el tipo de dato: {label}")
        

        for label in self.dict.keys():
            keys = list(self.dict[label].keys())
            keys.sort()
            self.dict[label] = {i: self.dict[label][i] for i in keys}    
            for i,key in enumerate(self.dict[label].keys()):
                self.dict[label][key] = i
        

        
        print(self.dict) #:))))))))))))))))))))))))))
        print(self.nominalAtributos) #:))))))))))))))))))))))
        
        
        
    
    # Devuelve el subconjunto de los datos cuyos �ndices se pasan como argumento
    def extraeDatos(self,idx):
        return self.datos.iloc[idx]


"""if __name__ == "__main__":
    
    datos = Datos("./datasets/heart.csv")
    print(datos.extraeDatos(0))"""