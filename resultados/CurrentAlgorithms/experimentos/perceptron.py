import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, file_name, N):
        self.df = pd.read_csv(file_name, header=None)
        self.A =  len(self.df.columns) - 2 # Atributos
        self.N = N # Neuronios
        self.embaralhar()
        self.separar_entrada_saida()
        self.separar_conjuntos()
        self.perceptron()

    def embaralhar(self):
        self.df = self.df.sample(frac=1)

    def separar_entrada_saida(self):
        self.entrada = pd.DataFrame(self.df[self.df.columns[1:self.A+1]].values)
        self.saida = self.df[self.df.columns[self.A+1]]
        self.saida = pd.DataFrame(self.saida.apply(self.transformar).values.tolist())

    def transformar(self, df):
        #if(df == 1):
        #    return [0, 0, 0, 1]

        #if(df == 2):
        #    return [0, 0, 1, 0]

        #if(df == 3):
        #    return [0, 0, 1, 1]

        #if(df == 4):
        #    return [0, 1, 0, 0]

        #if(df == 5):
        #    return [0, 1, 0, 1]

        #if(df == 6):
        #    return [0, 1, 1, 0]

        #if(df == 7):
        #    return [0, 1, 1, 1]

        if(df == 8):
            return [0, 1]
        #    return [1, 0, 0, 0]

       # if(df == 9):
       #     return [1, 0, 0, 1]

        #if(df == 10):
        #    return [1, 0, 1, 0]

        #if(df == 11):
        #    return [1, 0, 1, 1]

        if(df == 12):
            return [1, 1]
        #    return [1, 1, 0, 0]

        return [0, 0]

    def normalizar(self):
        pass

    def separar_conjuntos(self):
        tam = len(self.entrada)
        length_slice = int(round(tam / 3))
        self.T = length_slice #Numero de itens para teste
        self.M = tam - length_slice #Numero de itens para treinamento
        print("Items: ")
        print("   Teste: " + str(self.T))
        print("   Treino: " + str(self.M))

        self.conjunto_teste_entrada = self.entrada[:length_slice]
        self.conjunto_teste_saida = self.saida[:length_slice]
        
        self.conjunto_treino_entrada = self.entrada[length_slice:]
        self.conjunto_treino_saida = self.saida[length_slice:]


    def degrau(self, df):
        if (df >= 0):
            return 1
        return 0

    def perceptron(self):
        max_it = 200
        t = 0  # Iteracao (Epoca)
        b = np.zeros(self.N)  # Bias
        W = pd.DataFrame(np.zeros((self.N, self.A)))
        y = pd.DataFrame(np.zeros(self.N))
        e = pd.DataFrame(np.zeros(self.N))
        D = self.conjunto_treino_saida
        E = pd.DataFrame(np.zeros(max_it))
        alfa = 0.3
        corretos = 0       

        while (t < max_it):    
            for i in range(self.M):
                y = W.dot(self.conjunto_treino_entrada.iloc[i]).add(b).apply(self.degrau)
                e = D.iloc[i].subtract(y)
                E.iloc[t] += e.dot(e.T)
                update = e.values.reshape(self.N,1).dot(self.conjunto_treino_entrada.iloc[[i]]) * alfa
                W = W + update
                b = b + alfa * e

            t += 1
                
        for i in range(self.T):
            y = W.dot(self.conjunto_teste_entrada.iloc[i]).add(b).apply(self.degrau)
            if y.equals(self.conjunto_teste_saida.iloc[i]):
                corretos += 1

        self.W = W
        self.b = b

        print("Corretude: " + str((float(corretos)/self.T)*100))
       # plt.plot(E)
       # plt.show()

    def get_nota(self, entrada):
        result = self.W.dot(entrada).add(self.b).apply(self.degrau)
        if result.equals(pd.Series([1, 0, 0, 0])):
            print("Acorde G")
        elif result.equals(pd.Series([1, 1, 0, 0])):
            print("Acorde Si")
        else:
            print("Nao reconhecido")

                    

#p = Perceptron('/home/gabriela/projects/TCC/resultados/variacoes.csv', 2)

