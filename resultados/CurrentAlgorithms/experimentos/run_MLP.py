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


def generate_histogram_per_chord(acordes, total_mat_confusao, niter, max_it):
    for i, acorde_real in enumerate(acordes):
        filename = 'teste_mlp_{}_niter={}_alfa=0.001_epocas={}_{}.png'.format(acorde_real, niter, max_it, ''.join(acordes))
        title = r'Histograma acorde real {}'.format(acorde_real)
        predicao = total_mat_confusao[i]
        acordes_preditos = []
        for j, freq in enumerate(predicao):
            for k in range(int(freq)):
                acordes_preditos.append(acordes[j])
        save_hist(filename, title, acordes_preditos, 'Acordes preditos')


def teste_1(niter, treino, teste, acordes):
    """Testando acordes com sobreposicao de 1 nota: G e D"""
    print("Testando acordes com sobreposicao de 1 nota: G e D")
    corretudes = []
    mat_confusao_list = []
    df_train = pd.read_csv(treino)
    df_test = pd.read_csv(teste)
    x_train, y_train = df_train[df_train.columns[:9]], df_train[df_train.columns[-1]]
    x_test, y_test = df_test[df_test.columns[:9]], df_test[df_test.columns[-1]]
    csv_name = "teste_MLP_{}.csv".format(''.join(acordes))
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    max_it = 2500
    hidden_layer_sizes=(55,)
    for i in range(niter):
        clf = MLPClassifier(max_iter=max_it, hidden_layer_sizes=hidden_layer_sizes)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        mat_confusao = confusion_matrix(y_test, y_predict)
        mat_confusao_list.append(mat_confusao)
        corretude = clf.score(x_test, y_test) * 100
        corretudes.append(corretude)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [acordes])
        write_csv(csv_name, mat_confusao)
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    total_mat_confusao = np.asarray(mat_confusao_list).sum(axis=0)
    write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
    write_csv(csv_name, total_mat_confusao)
    generate_histogram_per_chord(acordes, total_mat_confusao, niter, max_it)
    filename = 'teste_MLP_{}.png'.format(''.join(acordes))
    title = r'Histograma {} $N={} $epoca={}'.format(' '.join(acordes), niter, max_it)
    save_hist(filename, title, corretudes)
    return corretudes
