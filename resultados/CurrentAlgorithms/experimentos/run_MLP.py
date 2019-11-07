import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from run import write_csv, save_hist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

NITER = 10

def write_csv(filename, results):
    import csv
    with open(filename, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(results)

def save_hist(filename, title, results, xlabel='Corretudes', ylabel='Número de vezes'):
    plt.clf()
    plt.hist(results)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)


def teste_1(niter=3):
    """Testando acordes com sobreposicao de 1 nota: G e D"""
    print("Testando acordes com sobreposicao de 1 nota: G e D")
    corretudes = []
    mat_confusao_list = []
    df_train = pd.read_csv('TmpDataSets/sol_re_9attr_ordenado_treino.csv')
    df_test = pd.read_csv('TmpDataSets/sol_re_9attr_ordenado_teste.csv')
    x_train, y_train = df_train[df_train.columns[:9]], df_train[df_train.columns[-1]]
    x_test, y_test = df_test[df_test.columns[:9]], df_test[df_test.columns[-1]]
    csv_name = "teste1_MLP.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    max_it = 200
    hidden_layer_sizes=(1,)
    for i in range(niter):
        clf = MLPClassifier(max_iter=max_it, hidden_layer_sizes=hidden_layer_sizes)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        mat_confusao = confusion_matrix(y_test, y_predict)
        mat_confusao_list.append(mat_confusao)
        corretude = clf.score(x_test, y_test) * 100
        corretudes.append(corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [["Sol", "Re"]])
        write_csv(csv_name, mat_confusao)
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    total_mat_confusao = np.asarray(mat_confusao_list).sum(axis=0)
    acordes = ["G", "D"]
    write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
    write_csv(csv_name, total_mat_confusao)
    for i, acorde_real in enumerate(acordes):
        filename = 'teste1_mlp_{}_niter={}_alfa=0.001_epocas={}_pesos_aleatorios.png'.format(acorde_real, niter, max_it)
        title = r'Histograma acorde real {}'.format(acorde_real)
        predicao = total_mat_confusao[i]
        acordes_preditos = []
        for j, freq in enumerate(predicao):
            for k in range(int(freq)):
                acordes_preditos.append(acordes[j])
        save_hist(filename, title, acordes_preditos, 'Acordes preditos')
    filename = 'teste1_Sol_Re_MLP.png'
    title = r'Histograma da execução do algoritmo com teste distribuido para G e D, $N={}'.format(niter)
    save_hist(filename, title, corretudes)
    return corretudes


def teste_2(niter):
    """Testando 3 acordes: G, C e D"""
    print("Testando 3 acordes: G, C e D")
    corretudes = []
    mat_confusao = []
    csv_name = "teste1_MLP.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        clf = MLPClassifier(max_iter=2500, hidden_layer_sizes=(55,))
        df = pd.read_csv('TmpDataSets/sol_re_9attr_ordenado.csv')
        mat_confusao.append(mat_confusao)
        corretudes.append(corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [["Sol", "Re"]])
        write_csv(csv_name, p.mat_confusao)
        write_csv(csv_name, [["Corretude", str(corretude)]])
    # print(corretudes)
    filename = 'teste1_Sol_Re_MLP.png'
    title = r'Histograma da execução do algoritmo com teste distribuido para G e D, $N={}'.format(niter)
    save_hist(filename, title, corretudes)
    return corretudes



def teste_3(niter):
    """Testando acordes com diferenca de 1 semiton: G e G#"""
    print("Testando acordes com diferenca de 1 semiton: G e G#")
    corretudes = []
    mat_confusao = []
    csv_name = "teste3_MLP.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/sol_sol#_9attr_ordenado.csv', 2)
        mat_confusao.append(mat_confusao)
        corretudes.append(corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [["Sol", "Sol#"]])
        write_csv(csv_name, p.mat_confusao)
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    filename = 'teste3_Sol_Sol#_MLP.png'
    title = r'Histograma da execução do algoritmo com teste distribuido para G e G#, $N={}'.format(niter)
    save_hist(filename, title, corretudes)
    return corretudes

def teste_4(niter): 
    """Testando acordes com diferenca de 1 tom: G e A"""
    print("Testando acordes com diferenca de 1 tom: G e A")
    corretudes = []
    mat_confusao = []
    csv_name = "teste4_MLP.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/sol_la_9attr_ordenado.csv', 2)
        mat_confusao.append(mat_confusao)
        corretudes.append(corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [["Sol", "La"]])
        write_csv(csv_name, p.mat_confusao)
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    filename = 'teste4_Sol_La_MLP.png'
    title = r'Histograma da execução do algoritmo com teste igualmente distribuido para G e A, $N={}'.format(niter)
    save_hist(filename, title, corretudes)
    return corretudes

def teste_5(niter):
    """Testando apenas para um acorde: G"""
    corretudes = []
    mat_confusao = []
    csv_name = "teste5_MLP.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/sol_misto_ordenado.csv', 1)
        mat_confusao.append(mat_confusao)
        corretudes.append(corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [["Sol", "NaoSol"]])
        write_csv(csv_name, p.mat_confusao)
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    filename = 'teste5_Sol_misto_MLP.png'
    title = r'Histograma da execução do algoritmo com teste distribuido para G e teste com acordes que nao sao G, $N={}'.format(niter)
    save_hist(filename, title, corretudes)

    return corretudes

def teste_6(niter, alfa=0.3, max_it=200):
    """Testando para 12 acordes maiores"""

    corretudes = []
    mat_confusao = []
    csv_name = "teste6_niter={}_alfa={}_epocas={}_pesos_aleatorios.csv".format(niter, alfa, ma_MLPx_it)
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/acordes_ordenados.csv', 12, alfa=alfa, max_it=max_it)
        corretudes.append(corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [p.acordes])
        write_csv(csv_name, p.mat_confusao[1:])
        mat_confusao.append(mat_confusao[1:])
        write_csv(csv_name, [["Corretude", str(corretude)]])
        #plt.plot(E)
        #plt.savefig("teste_6_erro_iter={}_alfa={}_epocas={}.png".format(i, alfa, max_it))

    print(corretudes)
    total_mat_confusao = np.asarray(mat_confusao).sum(axis=0)
    write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
    write_csv(csv_name, total_mat_confusao)
    for i, acorde_real in enumerate(acordes):
        filename = 'teste_6_{}_niter={}_alfa={}_epocas={}_pesos_aleatorios.png'.format(acorde_real, niter, alfa, max_it)
        title = r'Histograma acorde real {}'.format(acorde_real)
        predicao = total_mat_confusao[i][1:]
        acordes_preditos = []
        for j, freq in enumerate(predicao):
            for k in range(int(freq)):
                acordes_preditos.append(acordes[j])
        save_hist(filename, title, acordes_preditos, 'Acordes preditos')
    filename = 'teste6_acordes_ordenados_niter={}_alfa={}_epocas={}_pesos_aleatorios.png'.format(niter, alfa, max_it)
    title = r'Histograma da execução do algoritmo com os 12 acordes maiores, $N={}'.format(niter)
    save_hist(filename, title, corretudes)

    return corretudes

def teste_7(niter):
    """Testando para 12 acordes maiores com 3 atributos"""
    corretudes = []
    mat_confusao = []
    csv_name = "teste7_MLP.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/acordes_ordenados_attr3_selectkbest.csv', 12)
        corretudes.append(corretude)
        write_csv(csv_name, [p.acordes])
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, p.mat_confusao[1:])
        mat_confusao.append(mat_confusao[1:])
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    total_mat_confusao = np.asarray(mat_confusao).sum(axis=1)
    write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
    write_csv(csv_name, total_mat_confusao)
    filename = 'teste7_12acordes_attr3_selectkbest.png'
    title = r'Histograma da execução do algoritmo com os 12 acordes maiores 3 attr, $N={}'.format(niter)
    save_hist(filename, title, corretudes)

    return corretudes

def teste_8(niter):
    """Testando para 12 acordes maiores com 4 atributos"""
    corretudes = []
    mat_confusao = []
    csv_name = "teste8_MLP.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/acordes_ordenados_attr4_selectkbest.csv', 12)
        corretudes.append(corretude)
        write_csv(csv_name, [p.acordes])
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, p.mat_confusao[1:])
        mat_confusao.append(mat_confusao[1:])
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    total_mat_confusao = np.asarray(mat_confusao).sum(axis=1)
    write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
    write_csv(csv_name, total_mat_confusao)
    filename = 'teste8_12acordes_attr4_selectkbest.png'
    title = r'Histograma 12 acordes maiores 4 attr, $N={}'.format(niter)
    save_hist(filename, title, corretudes)

    return corretudes

def teste_9(niter):
    """Testando para 12 acordes maiores com 5 atributos"""
    corretudes = []
    mat_confusao = []
    csv_name = "teste9_MLP.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/acordes_ordenados_attr5_selectkbest.csv', 12)
        corretudes.append(corretude)
        write_csv(csv_name, [p.acordes])
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, p.mat_confusao[1:])
        mat_confusao.append(mat_confusao[1:])
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    total_mat_confusao = np.asarray(mat_confusao).sum(axis=1)
    write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
    write_csv(csv_name, total_mat_confusao)
    filename = 'teste7_12acordes_attr5_selectkbest.png'
    title = r'Histograma da execução do algoritmo 5 attr, $N={}'.format(niter)
    save_hist(filename, title, corretudes)

    return corretudes

def teste_10(niter):
    """Testando para 12 acordes maiores com 6 atributos"""
    corretudes = []
    mat_confusao = []
    csv_name = "teste10_MLP.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/acordes_ordenados_attr6_selectkbest.csv', 12)
        corretudes.append(corretude)
        write_csv(csv_name, [p.acordes])
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, p.mat_confusao[1:])
        mat_confusao.append(mat_confusao[1:])
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    total_mat_confusao = np.asarray(mat_confusao).sum(axis=1)
    write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
    write_csv(csv_name, total_mat_confusao)
    filename = 'teste7_12acordes_attr6_selectkbest.png'
    title = r'Histograma da execução do algoritmo 6 attr, $N={}'.format(niter)
    save_hist(filename, title, corretudes)

    return corretudes

def teste_11(niter):
    """Testando para 12 acordes maiores com 7 atributos"""
    corretudes = []
    mat_confusao = []
    csv_name = "teste11_MLP.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/acordes_ordenados_attr7_selectkbest.csv', 12)
        corretudes.append(corretude)
        write_csv(csv_name, [p.acordes])
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, p.mat_confusao[1:])
        mat_confusao.append(mat_confusao[1:])
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    total_mat_confusao = np.asarray(mat_confusao).sum(axis=1)
    write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
    write_csv(csv_name, total_mat_confusao)
    filename = 'teste7_12acordes_attr7_selectkbest.png'
    title = r'Histograma da execução do algoritmo 7 attr, $N={}'.format(niter)
    save_hist(filename, title, corretudes)

    return corretudes

def teste_12(niter):
    """Testando para 3 acordes maiores"""
    corretudes = []
    mat_confusao = []
    csv_name = "teste11_MLP.csv"
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    for i in range(niter):
        p = Perceptron('TmpDataSets/do_dosus_re.csv', 3)
        corretudes.append(corretude)
        write_csv(csv_name, [p.acordes])
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, p.mat_confusao[1:])
        mat_confusao.append(mat_confusao[1:])
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    total_mat_confusao = np.asarray(mat_confusao).sum(axis=1)
    write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
    write_csv(csv_name, total_mat_confusao)
    filename = 'teste7_12acordes_attr7_selectkbest.png'
    title = r'Histograma da execução do algoritmo 7 attr, $N={}'.format(niter)
    save_hist(filename, title, corretudes)

    return corretudes


def variar_parametros(self, teste_func, niter):
    alfa_list = [0.001, 0.01, 0.1]
    max_it_list = [200, 500, 1000]
    for alfa in alfa_list:
        for max_it in max_it_list:
            teste(niter=niter, alfa=alfa, max_it=max_it)
