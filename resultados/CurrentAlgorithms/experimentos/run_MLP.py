import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from run import write_csv, save_hist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

NITER = 10

def write_csv(filename, results):
    import csv
    with open(filename, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(results)

def save_hist(filename, title, results, xlabel='Acurácia', ylabel='Número de vezes'):
    plt.clf()
    if 'real' in title:
        plt.hist(results)
    else:
        plt.hist(results, range=(0, 100))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)


def generate_histogram_per_chord(acordes, total_mat_confusao, niter, max_it, info):
    for i, acorde_real in enumerate(acordes):
        filename = 'teste_mlp_{}_niter={}_alfa=0.001_epocas={}_{}.png'.format(acorde_real, niter, max_it, '_'.join(info.split()))
        title = r'Histograma acorde real {}'.format(acorde_real)
        predicao = total_mat_confusao[i]
        acordes_preditos = []
        for j, freq in enumerate(predicao):
            for k in range(int(freq)):
                acordes_preditos.append(acordes[j])
        save_hist(filename, title, acordes_preditos, 'Acordes preditos')


def teste_1(niter, treino, teste, info, acordes):
    """Testando acordes"""
    print("Testando acordes")
    corretudes = []
    mat_confusao_list = []
    df_train = pd.read_csv(treino)
    df_test = pd.read_csv(teste)
    x_train, y_train = df_train[df_train.columns[:5]], df_train[df_train.columns[-1]]
    x_test, y_test = df_test[df_test.columns[:5]], df_test[df_test.columns[-1]]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)
    
    csv_name = "teste_MLP_{}.csv".format('_'.join(info.split()))
    write_csv(csv_name, [["NITERACOES", str(niter)]])
    max_it = 2500
    hidden_layer_sizes=(9,)
    mlps = []
    iter_conv = []
    for i in range(niter):
        clf = MLPClassifier(verbose=True, max_iter=max_it)
        clf.fit(x_train_scaled, y_train)
        y_predict = clf.predict(x_test_scaled)
        mat_confusao = confusion_matrix(y_test, y_predict)
        mat_confusao_list.append(mat_confusao)
        corretude = clf.score(x_test_scaled, y_test) * 100
        corretudes.append(corretude)
        iter_conv.append(clf.n_iter_)
        mlps.append(clf)
        write_csv(csv_name, [["iter", i]])
        write_csv(csv_name, [acordes])
        write_csv(csv_name, mat_confusao)
        write_csv(csv_name, [["Corretude", str(corretude)]])
    print(corretudes)
    print(iter_conv)
    print([x.coefs_ for x in mlps])
    total_mat_confusao = np.asarray(mat_confusao_list).sum(axis=0)
    write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
    write_csv(csv_name, total_mat_confusao)
    generate_histogram_per_chord(acordes, total_mat_confusao, niter, max_it, info)
    filename = 'teste_MLP_scaled_{}.png'.format('_'.join(info.split()))
    title = r'Execução {} vezes do modelo {} $N={} $epoca={}'.format(info, niter, max_it)
    save_hist(filename, title, corretudes)
    # plt.plot(mlp.loss_curve_)
    # plt.show()

    print(corretudes)
    return corretudes


# acordes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# do_re_sol = ["C", "D", "G"]
# # teste_1(20, 'NewDatasets/acordes_CDG_treino.csv', 'NewDatasets/acordes_CDG_teste.csv', 'C D G 5 atributos scaled', do_re_sol)
# teste_1(20, 'NewDatasets/acordes_treino.csv', 'NewDatasets/acordes_teste.csv', '12 acordes 5 atributos padronizados', do_re_sol)

