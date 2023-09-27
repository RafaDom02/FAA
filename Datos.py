# -*- coding: utf-8 -*-

# coding: utf-8
import pandas as pd
import numpy as np

import os

class Datos:

    diccionarios = {}
    nominalAtributos = []
    datos = None

    # Constructor: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionarios
    def __init__(self, nombreFichero):  
        # Carga el fichero de datos
        self.datos = pd.read_csv(nombreFichero)

        #get a dictionary of labels from the csv file
        aux_list = self.datos.to_dict(orient="records")

        keys = list(aux_list[0].keys())
        #añadir las keys al diccionario
        for i in range(len(keys)):
            self.diccionarios[keys[i]] = {}
        
        for i in range(len(aux_list)):
            for label in self.diccionarios.keys():
                if aux_list[i][label] not in self.diccionarios[label]:
                    if  not (type(aux_list[i][label]) == int or type(aux_list[i][label]) == float) or\
                        label.casefold() == "Class".casefold():

                        if i == 0:
                            self.nominalAtributos.append(True)
                        if aux_list[i][label] not in self.diccionarios[label].values():
                            self.diccionarios[label].update({str(aux_list[i][label]): None})
                    elif (type(aux_list[i][label]) == int or type(aux_list[i][label])):
                            if i == 0:
                                self.nominalAtributos.append(False)
                    else:
                        raise ValueError(f"Error en el tipo de dato: {label}")
        

        for label in self.diccionarios.keys():
            s_keys = list(self.diccionarios[label].keys())
            s_keys.sort()
            self.diccionarios[label] = {i: self.diccionarios[label][i] for i in s_keys}    
            for i,key in enumerate(self.diccionarios[label].keys()):
                self.diccionarios[label][key] = i
        

        for key in keys:
            for dict_key in self.diccionarios[key].keys():
               self.datos[key] = self.datos[key].replace([dict_key], self.diccionarios[key][dict_key])

        aux_dict_type = {}
        for key in keys:
            aux_dict_type[key] = float

        self.datos=self.datos.astype(aux_dict_type)
    
    # Devuelve el subconjunto de los datos cuyos �ndices se pasan como argumento
    def extraeDatos(self,idx):
        return [self.datos.iloc[i] for i in idx]