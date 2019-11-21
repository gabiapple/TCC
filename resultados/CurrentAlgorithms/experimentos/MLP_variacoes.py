from run_MLP import write_csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix



acordes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

treino = 'NewDatasets/acordes_treino_sem_limitar_faixa.csv'
teste = 'NewDatasets/acordes_teste_sem_limitar_faixa.csv'

df_train = pd.read_csv(treino)
df_test = pd.read_csv(teste)

num_atributos = 9

X_train, y_train = df_train[df_train.columns[:num_atributos]].values, df_train[df_train.columns[-1]]
X_test, y_test = df_test[df_test.columns[:num_atributos]].values, df_test[df_test.columns[-1]]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


def variando_camadas_escondidas():
    max_score = 0
    max_score_neurons = 0
    csv_name = "test_neuronios.csv"
    range_neuronios = list(range(100, 10, -10))
    for i in range(5):
        mlps = []
        corretudes = []
        for neuronios in range_neuronios:
            mlp = MLPClassifier(hidden_layer_sizes=(neuronios,), max_iter=1, alpha=1e-4,
                            solver='adam', tol=1e-4, random_state=i)
            N_TRAIN_SAMPLES = X_train.shape[0]
            N_EPOCHS = 2000
            N_BATCH = min(128, N_TRAIN_SAMPLES)
            N_CLASSES = np.unique(y_train)

            scores_train = []
            scores_test = []

            # EPOCH
            epoch = 0
            print(i)
            while epoch < N_EPOCHS:
                # print('epoch: ', epoch)
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

                if mlp._no_improvement_count > mlp.n_iter_no_change:
                    break

                epoch += 1
            if score_test > max_score:
                max_score = score_test
                max_score_execution = neuronios

            mlps.append(mlp)
            
            # PREDICT
            y_predict = mlp.predict(X_test)
            mat_confusao = confusion_matrix(y_test, y_predict)
            # mat_confusao_list.append(mat_confusao)

            corretude = score_test * 100
            corretudes.append(corretude)
            write_csv(csv_name, [["neuronios", neuronios]])
            write_csv(csv_name, [acordes])
            write_csv(csv_name, mat_confusao)
            write_csv(csv_name, [["Corretude", str(corretude)]])
            # write_csv(csv_name, [["Relatorio classificador"]])

            report = classification_report(y_test, y_predict, output_dict=True)

            df_report = pd.DataFrame(report).transpose()
            # with open(csv_name, 'a') as f:
            #     df_report.to_csv(f)
            

            """ Plot train_test"""
            # mlps.append(mlp)
            # plt.figure()
            # plt.plot(scores_train, color='green', alpha=0.8, label='Treino')
            # plt.plot(scores_test, color='magenta', alpha=0.8, label='Teste')
            # plt.title("Acurácia ao longo das épocas", fontsize=14)
            # plt.xlabel('Épocas')
            # plt.legend(loc='upper left')
            # plt.show()
            # plt.savefig('acuracia_treino_teste_execucao{}_{}neuronios_escondidos'.format(i, neuronios))
            # plt.figure()
            # plt.title("Função de perda ao longo das épocas", fontsize=14)
            # plt.xlabel('Épocas')
            # plt.plot(mlp.loss_curve_)
            # plt.show()
            # plt.savefig('funcao_perda_execucao{}_9attr'.format(i))
        write_csv(csv_name, [["Tabela de acuracias - latex format", "execucao {}".format(str(i))]])
        n_iter_list = [mlp.n_iter_ for mlp in mlps]
    
        tabela_acuracias = []
    
        for j, qtd_neuronios in enumerate(range_neuronios):
            execucao = str(i)
            neuronio_info = str(qtd_neuronios)
            acuracia = str(round(corretudes[j], 2))
            epocas = str(n_iter_list[j])
            row = "\#{execucao} && {neuronios} && {epocas} && {acuracia}".format(execucao=execucao, neuronios=neuronio_info, epocas=epocas, acuracia=acuracia)
            tabela_acuracias.append([row])
        write_csv(csv_name, tabela_acuracias)

    # total_mat_confusao = np.asarray(mat_confusao_list).sum(axis=0)
    # write_csv(csv_name, [["MATRIZ DE CONFUSAO ACUMULADA"], str(niter)])
    # write_csv(csv_name, total_mat_confusao)
    # info = '12 chords {} attr'.format(num_atributos)
    # mean_n_iter = int(np.mean(n_iter_list))
    # generate_histogram_per_chord(acordes, total_mat_confusao, niter, mean_n_iter, info)
    # filename = '{}/teste_MLP_{}.png'.format(test_directory, '_'.join(info.split()))
    # title = r'Execução {} vezes do modelo $maxEpocas={}'.format(niter, N_EPOCHS)
    # save_hist(filename, title, corretudes)
    # write_csv(csv_name, [["Melhor execução: {}".format(max_score_execution)]])
    # write_csv(csv_name, [["DADOS DO MODELO"]])
    # write_csv(csv_name, [["epoca que convergiu", mlps[max_score_execution].n_iter_]])
    # # write_csv(csv_name, [["Coefs"]])
    # # write_csv(csv_name, mlps[max_score_execution].coefs_)
    # # write_csv(csv_name, [["Intercepts"]])
    # # write_csv(csv_name, mlps[max_score_execution].intercepts_)
    # # Monta tabela de acuracias
    # write_csv(csv_name, [["Tabela de acuracias - latex format"]])
    # tabela_acuracias = []
    # for i, neuronios in enumerate(range_neuronios):
    #     name = "#" + str(neuronios)
    #     tabela_acuracias.append(["\#" + str(i) + " & {} //".format(str(round(corretudes[i], 2)))])

    # write_csv(csv_name, tabela_acuracias)

variando_camadas_escondidas()