import csv
import numpy as np
import pandas as pd


class DataSet:
    def __init__(self, csv_to_read, csv_to_write):
        self.read_csv(csv_to_read)
        self.write_csv(csv_to_write)

    def read_csv(self, filename):
        data = []
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                instancia = []
                inicio = 1
                fim = len(row) - 1
                frequencias = row[inicio:fim]
                frequencias.sort() # ordena as frequencias
                instancia.append(row[0]) # index
                instancia.extend(frequencias) # atributos
                instancia.append(row[fim]) # classe
                data.append(instancia)

        self.df = pd.DataFrame(data)
        self.data = data
    
    def write_csv(self, filename):
        with open(filename, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerows(self.data)



obj = DataSet('TmpDataSets/acordes.csv', 'TmpDataSets/acordes_ordenados.csv')
