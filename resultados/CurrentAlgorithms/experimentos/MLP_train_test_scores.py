import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

from run_MLP import write_csv, save_hist, generate_histogram_per_chord

np.random.seed(1)

acordes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# treino = 'NewDatasets/acordes_treino.csv'
# teste = 'NewDatasets/acordes_teste.csv'

treino = 'NewDatasets/acordes_treino_sem_limitar_faixa.csv'
teste = 'NewDatasets/acordes_teste_sem_limitar_faixa.csv'

df_train = pd.read_csv(treino)
df_test = pd.read_csv(teste)

num_atributos = 9

x_train, y_train = df_train[df_train.columns[:num_atributos]], df_train[df_train.columns[-1]]
x_test, y_test = df_test[df_test.columns[:num_atributos]], df_test[df_test.columns[-1]]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.fit_transform(x_test)

""" Home-made mini-batch learning
    -> not to be used in out-of-core setting!
"""

mlps = []
max_score = 0
max_score_execution = 0
corretudes = []
mat_confusao_list = []

# csv_name = "teste_MLP_12acordes_5attr_100neuronios.csv"
   
niter = 20

for i in range(niter):
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                    solver='adam', verbose=True, tol=1e-4, random_state=i)
    N_TRAIN_SAMPLES = X_train.shape[0]
    N_EPOCHS = 2000
    N_BATCH = min(128, N_TRAIN_SAMPLES)
    N_CLASSES = np.unique(y_train)

    scores_train = []
    scores_test = []

    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        # SHUFFLING
        random_perm = np.random.permutation(X_train.shape[0])
        mini_batch_index = 0
        while True:
            # MINI-BATCH
            indices = random_perm[mini_batch_index:mini_batch_index + N_BATCH]
            mlp.partial_fit(X_train[indices], y_train[indices], classes=N_CLASSES)
            mini_batch_index += N_BATCH

            if mini_batch_index >= N_TRAIN_SAMPLES:
                break

        # SCORE TRAIN
        score_train = mlp.score(X_train, y_train)
        scores_train.append(score_train)

        # SCORE TEST
        score_test = mlp.score(X_test, y_test)
        scores_test.append(score_test)

        epoch += 1
    if score_test > max_score:
        max_score = score_test
        max_score_execution = i
    
    # PREDICT
    y_predict = mlp.predict(X_test)
    mat_confusao = confusion_matrix(y_test, y_predict)
    mat_confusao_list.append(mat_confusao)

    corretude = score_test * 100
    corretudes.append(corretude)
    # write_csv(csv_name, [["iter", i]])
    # write_csv(csv_name, [acordes])
    # write_csv(csv_name, mat_confusao)
    # write_csv(csv_name, [["Corretude", str(corretude)]])
    # write_csv(csv_name, [["Relatorio classificador"]])

    # report = classification_report(y_test, y_predict, output_dict=True)

    # df_report = pd.DataFrame(report).transpose()
    # with open(csv_name, 'a') as f:
    #     df_report.to_csv(f)
    

    """ Plot train_test"""
    mlps.append(mlp)
    # plt.figure()
    # plt.plot(scores_train[:mlp.n_iter_], color='green', alpha=0.8, label='Treino')
    # plt.plot(scores_test[:mlp.n_iter_], color='magenta', alpha=0.8, label='Teste')
    # plt.title("Acurácia ao longo das épocas", fontsize=14)
    # plt.xlabel('Épocas')
    # plt.legend(loc='upper left')
    # # plt.show()
    # plt.savefig('acuracia_treino_teste_execucao{}'.format(i))
    plt.figure()
    plt.title("Função de perda ao longo das épocas", fontsize=14)
    plt.xlabel('Épocas')
    plt.plot(mlp.loss_curve_)
    plt.savefig('funcao_perda_execucao{}_9attr'.format(i))

# total_mat_confusao = np.asarray(mat_confusao_list).sum(axis=0)
# write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
# write_csv(csv_name, total_mat_confusao)
# info = '12chords'
# generate_histogram_per_chord(acordes, total_mat_confusao, niter, N_EPOCHS, info)
# filename = 'teste_MLP_scaled_{}.png'.format('_'.join(info.split()))
# title = r'Execução {} vezes do modelo $maxEpocas={}'.format(niter, N_EPOCHS)
# save_hist(filename, title, corretudes)
# write_csv(csv_name, [["Melhor execução: {}".format(max_score_execution)]])
# write_csv(csv_name, [["DADOS DO MODELO"]])
# write_csv(csv_name, [["epoca que convergiu", mlps[max_score_execution].n_iter_]])
# write_csv(csv_name, [["Coefs"]])
# write_csv(csv_name, mlps[max_score_execution].coefs_)
# write_csv(csv_name, [["Intercepts"]])
# write_csv(csv_name, mlps[max_score_execution].intercepts_)

    