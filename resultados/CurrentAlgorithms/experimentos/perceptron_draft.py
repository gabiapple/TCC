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
        self.separar_conjuntos_novo()
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
        if self.N == 1:
            if df == 8:
                return 1
            return 0
            
        # data = [0 for i in range(self.N)]
        # data[df - 1] = 1
        # self.get_acorde[tuple(data)] = Perceptron.acordes[df - 1]

        if df == 8:
            return [1, 0]
        
        return [0, 1]
 
        # return data

    def inv_transformar(self, df):
        if self.N == 1:
            if df == 1:
                return 0
            return 1
            
        # data = [0 for i in range(self.N)]
        # data[df - 1] = 1
        # self.get_acorde[tuple(data)] = Perceptron.acordes[df - 1]

        if df.values.tolist() == [1,0]:
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

    def separar_conjuntos_novo(self):
        tam = len(self.entrada)
        length_slice = int(round(tam / 3))
        self.T = length_slice #Numero de itens para teste
        self.M = tam - length_slice #Numero de itens para treinamento
        # col_classe = self.saida.columns[-1]
        classes = self.saida.drop_duplicates().values.tolist()
        num_classes = len(classes)
        itens_por_classe = int(self.T / num_classes)
        # self.conjunto_teste_entrada
        instancias_id = []

        for classe in classes:
            instancias = self.saida[self.saida == classe].dropna()
            instancias_id.extend(instancias.sample(n=int(itens_por_classe)).index.to_list())

        if len(instancias_id) < self.T:
            instancias_id.append(self.saida[~self.saida.index.isin(instancias_id)] 
                .sample(n=1).index.values[0])
        
        # conjunto_teste_entrada.loc[instancias.sample(n=int(itens_por_classe)).index]

        # cols_atributos = [i for i in range(1, self.A+1)]

        self.conjunto_teste_entrada = pd.DataFrame(self.entrada.loc[instancias_id].values)
        self.conjunto_teste_saida = pd.DataFrame(self.saida.loc[instancias_id].values)
        
        self.conjunto_treino_entrada = pd.DataFrame(self.entrada.loc[~self.df.index.isin(instancias_id)].values)
        self.conjunto_treino_saida = pd.DataFrame(self.saida.loc[~self.df.index.isin(instancias_id)].values)


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
            predict_y.append(y.tolist())
            real_y.append(self.conjunto_teste_saida.iloc[i].tolist())

            # acorde_teste_y = self.get_acorde.get(tuple(self.conjunto_teste_saida.iloc[i]), 'Nao reconhecido')
            # acorde_predict_y = self.get_acorde.get(tuple(y), 'Nao reconhecido')
           
            # teste_y.append(acorde_teste_y)
            # predict_y.append(acorde_predict_y)
        predict_y = pd.DataFrame(predict_y).apply(self.inv_transformar, axis=1).values.tolist()
        real_y = pd.DataFrame(real_y).apply(self.inv_transformar, axis=1).values.tolist()
        # import pdb; pdb.set_trace()
       
        self.avaliacao = Resultado(np.asarray(real_y), np.asarray(predict_y))
        self.mat_confusao = self.avaliacao.mat_confusao
        
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
    corretudes = []
    mat_confusao = []
    csv_name = "teste1.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/sol_re_9attr_ordenado.csv', 2)
        mat_confusao.append(p.mat_confusao)
        corretudes.append(p.corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [["Sol", "Re"]])
        write_csv(csv_name, p.mat_confusao)
        write_csv(csv_name, [["Corretude", str(p.corretude)]])
    # print(corretudes)
    filename = 'teste1_Sol_Re_igualmente_distribuido.png'
    title = r'Histograma da execução do algoritmo com teste distribuido para G e D, $N=20$'
    save_hist(filename, title, corretudes)
    return corretudes

def teste_2(niter):   
    """Testando acordes com sobreposicao de 2 notas: G e B"""
    print("Testando acordes com sobreposicao de 2 notas: G e B")
    corretudes = []
    mat_confusao = []
    csv_name = "teste2.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/sol_si_9attr_ordenado.csv', 2)
        mat_confusao.append(p.mat_confusao)
        corretudes.append(p.corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [["Sol", "Si"]])
        write_csv(csv_name, p.mat_confusao)
        write_csv(csv_name, [["Corretude", str(p.corretude)]])
    print(corretudes)
    filename = 'teste2_Sol_Si_igualmente_distribuido.png'
    title = r'Histograma da execução do algoritmo com teste distribuido para G e B, $N=20$'
    save_hist(filename, title, corretudes)
    return corretudes

def teste_3(niter):   
    """Testando acordes com diferenca de 1 semiton: G e G#"""
    print("Testando acordes com diferenca de 1 semiton: G e G#")
    corretudes = []
    mat_confusao = []
    csv_name = "teste3.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/sol_sol#_9attr_ordenado.csv', 2)
        mat_confusao.append(p.mat_confusao)
        corretudes.append(p.corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [["Sol", "Sol#"]])
        write_csv(csv_name, p.mat_confusao)
        write_csv(csv_name, [["Corretude", str(p.corretude)]])
    print(corretudes)
    filename = 'teste3_Sol_Sol#_igualmente_distribuido.png'
    title = r'Histograma da execução do algoritmo com teste distribuido para G e G#, $N=20$'
    save_hist(filename, title, corretudes)
    return corretudes

def teste_4(niter): 
    """Testando acordes com diferenca de 1 tom: G e A"""
    print("Testando acordes com diferenca de 1 tom: G e A")
    corretudes = []
    mat_confusao = []
    csv_name = "teste4.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/sol_la_9attr_ordenado.csv', 2)
        mat_confusao.append(p.mat_confusao)
        corretudes.append(p.corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [["Sol", "La"]])
        write_csv(csv_name, p.mat_confusao)
        write_csv(csv_name, [["Corretude", str(p.corretude)]])
    print(corretudes)
    filename = 'teste4_Sol_La_igualmente_distribuido.png'
    title = r'Histograma da execução do algoritmo com teste igualmente distribuido para G e A, $N=20$'
    save_hist(filename, title, corretudes)
    return corretudes

def teste_5(niter):
    """Testando apenas para um acorde: G"""
    corretudes = []
    mat_confusao = []
    csv_name = "teste5.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/sol_misto_ordenado.csv', 1)
        mat_confusao.append(p.mat_confusao)
        corretudes.append(p.corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [["Sol", "NaoSol"]])
        write_csv(csv_name, p.mat_confusao)
        write_csv(csv_name, [["Corretude", str(p.corretude)]])
    print(corretudes)
    filename = 'teste5_Sol_misto_igualmente_distribuido.png'
    title = r'Histograma da execução do algoritmo com teste distribuido para G e teste com acordes que nao sao G, $N=20$'
    save_hist(filename, title, corretudes)

    return corretudes
    

# results = [["Nome do Teste"]]

# for i in range(1, NITER):
#     results[0].append("Iteracao {}".format(str(i)))

# results.append(["teste_1_ordenado"])
# results[-1].extend(teste_1(NITER))
# results.append(["teste_2_ordenado"])
# results[-1].extend(teste_2(NITER))
# results.append(["teste_3_ordenado"])
# results[-1].extend(teste_3(NITER))
# results.append(["teste_4_ordenado"])
# results[-1].extend(teste_4(NITER))

# results = ["teste_5"]
# corretudes, corretudes_validacao = teste_5(NITER)
# print(corretudes, corretudes_validacao)
# results.extend(teste_5(NITER))
# print(results)
# write_csv("results_9attr_dataset_ordenado.csv", [results])
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

# p = Perceptron('TmpDataSets/sol_re_9attr_ordenado.csv', 2)
teste_5(NITER)