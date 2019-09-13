import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from avaliacao import Resultado

class Perceptron:
    acordes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    def __init__(self, file_name, N, add_conjunto_teste=False, conjunto_teste=[]):
        self.df = pd.read_csv(file_name, header=None)
        self.A =  len(self.df.columns) - 2 # Atributos
        self.N = N # Neuronios
        self.get_acorde = {}
        self.get_acorde[tuple(0 for i in range(self.N))] = "Nao reconhecido"
        self.embaralhar()
        self.separar_entrada_saida()
        self.separar_conjuntos()
        if add_conjunto_teste:
            self.adicionar_conjunto_teste(conjunto_teste)
        self.perceptron()

    def embaralhar(self):
        self.df = self.df.sample(frac=1)

    def separar_entrada_saida(self):
        self.entrada = pd.DataFrame(self.df[self.df.columns[1:self.A+1]].values)
        self.saida = self.df[self.df.columns[self.A+1]]
        self.saida = pd.DataFrame(self.saida.apply(self.transformar).values.tolist())

    def transformar(self, df):
        # data = [0 for i in range(self.N)]
        # data[df - 1] = 1
        # self.get_acorde[tuple(data)] = Perceptron.acordes[df - 1]

        if df != 8:
            return 0
        
        return 1

        # return data

    def normalizar(self):
        pass

    def separar_conjuntos(self):
        tam = len(self.entrada)
        length_slice = int(round(tam / 3))
        self.T = length_slice #Numero de itens para teste
        self.M = tam - length_slice #Numero de itens para treinamento

        self.conjunto_teste_entrada = self.entrada[:length_slice]
        self.conjunto_teste_saida = self.saida[:length_slice]
        
        self.conjunto_treino_entrada = self.entrada[length_slice:]
        self.conjunto_treino_saida = self.saida[length_slice:]

    def adicionar_conjunto_teste(self, conjunto):
        df = pd.DataFrame(conjunto)
        new_teste_entrada = pd.DataFrame(df[df.columns[1:self.A+1]].values)
        new_teste_saida = df[df.columns[self.A+1]]
        new_teste_saida = pd.DataFrame(new_teste_saida.apply(self.transformar).values.tolist())

        if len(df.columns) - 2 != self.A:
            print("Novo conjunto invalido pois numero de atributos nao condiz com o existente")
            exit(-1)
        self.conjunto_teste_entrada = pd.concat([self.conjunto_teste_entrada, new_teste_entrada])
        self.conjunto_teste_saida = pd.concat([self.conjunto_teste_saida, new_teste_saida])
        self.T += len(new_teste_entrada)

    def degrau(self, df):
        if (df >= 0):
            return 1
        return 0

    def perceptron(self):
        print("Items: ")
        print("   Teste: " + str(self.T))
        print("   Treino: " + str(self.M))
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
                # import pdb; pdb.set_trace()
                y = W.dot(self.conjunto_treino_entrada.iloc[i]).add(b).apply(self.degrau)
                e = D.iloc[i].subtract(y)
                E.iloc[t] += e.dot(e.T)
                update = e.values.reshape(self.N,1).dot(self.conjunto_treino_entrada.iloc[[i]]) * alfa
                W = W + update
                b = b + alfa * e

            t += 1

        real_y = []
        predict_y = []
                
        for i in range(self.T):
            y = W.dot(self.conjunto_teste_entrada.iloc[i]).add(b).apply(self.degrau)
              
            if y.equals(self.conjunto_teste_saida.iloc[i]):
                corretos += 1

            predict_y.append(y[0])
            real_y.append(self.conjunto_teste_saida.iloc[i][0])

            # acorde_teste_y = self.get_acorde.get(tuple(self.conjunto_teste_saida.iloc[i]), 'Nao reconhecido')
            # acorde_predict_y = self.get_acorde.get(tuple(y), 'Nao reconhecido')
           
            # teste_y.append(acorde_teste_y)
            # predict_y.append(acorde_predict_y)

        self.avaliacao = Resultado(np.asarray(real_y), np.asarray(predict_y))

        self.W = W
        self.b = b

        self.corretude = (float(corretos)/self.T)*100
        print("Corretude: " + str(self.corretude))
        # plt.plot(E)
        # plt.show()

    # def get_acorde(self, entrada):
    #     result = self.W.dot(entrada).add(self.b).apply(self.degrau)
    #     print(result)
    #     if 1 in result:
    #         index = result[result == 1]
    #         print(dir(index))
    #         import pdb; pdb.set_trace()
    #         if isinstance(index, pd.RangeIndex):
    #             print("Acorde nao reconhecido")
    #             return "Acorde nao reconhecido"
    #         print("Acorde: " + Perceptron.acordes[index])
    #         return Perceptron.acordes[index]
    #     print("Acorde nao reconhecido")
    #     return "Acorde nao reconhecido"

    def validar(self, teste):
        df = pd.DataFrame(teste)
        if len(df.columns) - 2 != self.A:
            print("Novo conjunto invalido pois numero de atributos nao condiz com o existente")
            exit(-1)
        teste_entrada = pd.DataFrame(df[df.columns[1:self.A+1]].values)
        teste_saida = df[df.columns[self.A+1]]
        teste_saida = pd.DataFrame(teste_saida.apply(self.transformar).values.tolist())
        corretos = 0
        predict_y = []
        real_y = []
        for i in range(len(teste)):
            y = self.W.dot(teste_entrada.iloc[i]).add(self.b).apply(self.degrau)
            if y.equals(teste_saida.iloc[i]):
                corretos += 1
            predict_y.append(y[0])
            real_y.append(teste_saida.iloc[i][0])

        r = Resultado(np.asarray(real_y), np.asarray(predict_y))
        print(r.mat_confusao)
        self.mat_confusao = r.mat_confusao
        self.corretude_validacao = (float(corretos)/len(teste))*100
        print("Corretude: " + str(self.corretude_validacao))

    def add_teste(self, entrada, saida):
        new_teste = pd.DataFrame(entrada)
        self.conjunto_teste_entrada.append(new_teste)
        self.conjunto_teste_saida.append(self.transformar(saida))
        self.T += 1

    def predict(data):
        return W.dot(data).add(b).apply(self.degrau)

NITER = 20

def write_csv(filename, results):
    import csv
    with open(filename, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(results)

def save_hist(filename, title, results):
    plt.clf()
    plt.hist(results)
    plt.xlabel('Corretudes')
    plt.ylabel('Número de vezes')
    plt.title(title)
    plt.savefig(filename)


def teste_1(niter):
    """Testando acordes com sobreposicao de 1 nota: G e D"""
    print("Testando acordes com sobreposicao de 1 nota: G e D")
    # corretudes = []
    # for i in range(niter):
    #     p = Perceptron('Sol_Re.csv', 2)
    #     corretudes.append(p.corretude)
    # corretudes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 83.33333333333334, 100.0, 100.0, 100.0, 83.33333333333334, 100.0, 100.0, 100.0, 100.0, 100.0]
    # filename = 'teste1_Sol_Re.png'
    # title = r'Histograma da execução do algoritmo para G e D, $N=20$'
    # save_hist(filename, title, corretudes)
    return corretudes

def teste_2(niter):   
    """Testando acordes com sobreposicao de 2 notas: G e B"""
    print("Testando acordes com sobreposicao de 2 notas: G e B")
    # corretudes = []
    # for i in range(niter):
    #     p = Perceptron('Sol_Si.csv', 2)
    #     corretudes.append(p.corretude)
    corretudes = [100.0, 100.0, 66.66666666666666, 83.33333333333334, 83.33333333333334, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 83.33333333333334, 100.0, 100.0, 83.33333333333334, 100.0, 100.0, 100.0, 100.0, 100.0]
    filename = 'teste2_Sol_Si.png'
    title = r'Histograma da execução do algoritmo para G e B, $N=20$'
    save_hist(filename, title, corretudes)
    return corretudes

def teste_3(niter):   
    """Testando acordes com diferenca de 1 semiton: G e G#"""
    print("Testando acordes com diferenca de 1 semiton: G e G#")
    # corretudes = []
    # for i in range(niter):
    #     p = Perceptron('Sol_Sol#.csv', 2)
    #     corretudes.append(p.corretude)
    corretudes = [66.66666666666666, 83.33333333333334, 100.0, 100.0, 83.33333333333334, 83.33333333333334, 83.33333333333334, 100.0, 100.0, 100.0, 83.33333333333334, 100.0, 100.0, 50.0, 83.33333333333334, 83.33333333333334, 100.0, 100.0, 83.33333333333334, 83.33333333333334]
    filename = 'teste3_Sol_Sol#.png'
    title = r'Histograma da execução do algoritmo para G e G#, $N=20$'
    save_hist(filename, title, corretudes)
    return corretudes

def teste_4(niter): 
    """Testando acordes com diferenca de 1 tom: G e A"""
    print("Testando acordes com diferenca de 1 tom: G e A")
    # corretudes = []
    # for i in range(niter):
    #     p = Perceptron('Sol_La.csv', 2)
    #     corretudes.append(p.corretude)
    corretudes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 66.66666666666666, 100.0, 100.0, 100.0, 100.0, 83.33333333333334, 100.0, 100.0, 83.33333333333334, 100.0, 66.66666666666666, 100.0]
    filename = 'teste4_Sol_La.png'
    title = r'Histograma da execução do algoritmo para G e A, $N=20$'
    save_hist(filename, title, corretudes)
    return corretudes

def teste_5(niter):
    """Testando apenas para um acorde: G"""
    corretudes = []
    corretudes_validacao = []
    acordes_nao_sol = [
        [10, 220,221,166,222,248,330,167,209,222, 10],
        [11, 147,221,148,221,233,296,187,192,221, 3],
        [12, 248,247,166,209,249,332,166,206,216, 5]
    ]
    for i in range(1):
        p = Perceptron('Sol.csv', 1, add_conjunto_teste=True, conjunto_teste=acordes_nao_sol)
        print("Matriz de confusao", p.avaliacao.mat_confusao)
        print("Acuracia", p.avaliacao.acuracia)
        print("Precisao", p.avaliacao.precisao)
        print("Revocacao", p.avaliacao.revocacao)
        # print(p.avaliacao.mat_confusao, p.avaliacao.acuracia, p.avaliacao.precisao, p.avaliacao.revocacao)
        corretudes.append(p.corretude)
    #     corretudes_validacao.append(p.corretude_validacao)
    # print(corretudes, corretudes_validacao)
    # filename = 'teste5_Sol_treino_teste_mesmo_acorde.png'
    # title = r'Histograma da execução do algoritmo para G, treino e teste apenas com G, $N=20$'
    # save_hist(filename, title, corretudes)
    # corretudes = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    # corretudes_validacao = []
    # for i in range(niter):
    #     p = Perceptron('Sol.csv', 1)
    #     p.validar(acordes_nao_sol)
    #     corretudes_validacao.append(p.corretude_validacao)
    # print(corretudes_validacao)
    # filename = 'teste5_Sol_teste_naoSol.png'
    # title = r'Histograma da execução do algoritmo para G e teste com acordes que nao sao G, $N=20$'
    # save_hist(filename, title, corretudes)

    return corretudes
    

# results = [["Nome do Teste"]]

# for i in range(1, NITER):
#     results[0].append("Iteracao {}".format(str(i)))

# results.append(["teste_1"])
# results[-1].extend(teste_1(NITER))
# results.append(["teste_2"])
# results[-1].extend(teste_2(NITER))
# results.append(["teste_3"])
# results[-1].extend(teste_3(NITER))
# results.append(["teste_4"])
# results[-1].extend(teste_4(NITER))

results = ["teste_5"]
corretudes, corretudes_validacao = teste_5(NITER)
print(corretudes, corretudes_validacao)
# results.extend(teste_5(NITER))
# print(results)
# write_csv("results.csv", [results])
# results.extend([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
# print(results)
# write_csv("results.csv", [results])

# print(results)

# write_csv('results.csv', results)

# p = Perceptron('Sol_Re.csv', 12)
# print(p.W)
# print(p.get_acorde([220,221,166,222,248,330,167,209,222]))


# p = Perceptron('~/projects/TCC/resultados/CurrentAlgorithms/Datasets/acordesMaiores.csv', 12)
# avaliar_resultado = Resultado()

