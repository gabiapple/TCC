import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from plot_learning_curve import plot_learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from run import write_csv, save_hist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler


def get_params(x_train, y_train):
    from sklearn.model_selection import GridSearchCV
    param_grid = [
        {
            'activation' : ['logistic', 'tanh', 'relu'],
            'solver' : ['sgd', 'adam'],
            'hidden_layer_sizes': [(i+1,) for i in range(0, 101, 12)],
            'max_iter': [2000, 4000, 8000]
        }
       ]
    clf2 = GridSearchCV(MLPClassifier(verbose=True), param_grid, cv=3,
                        scoring='accuracy')
    clf2.fit(x_train, y_train)
    return clf2, clf2.best_params_

def plot_validation_curve(X,y, param, param_range):
    # Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    from sklearn.model_selection import validation_curve

    train_scores, test_scores = validation_curve(MLPClassifier(), X, y, param,
                                               param_range,
                                               cv=3)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    title = "MPL"
    plt.title("Validation Curve with MPL")
    plt.xlabel(r"${}$".format(param))
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    # plt.fill_between(param_range, train_scores_mean - train_scores_std,
    #                 train_scores_mean + train_scores_std, alpha=0.2,
    #                 color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    # plt.fill_between(param_range, test_scores_mean - test_scores_std,
    #                 test_scores_mean + test_scores_std, alpha=0.2,
    #                 color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


def example_plot_learning_curve(X, y):
    # Reference: https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/
    train_sizes, train_scores, test_scores = learning_curve(MLPClassifier(max_iter=2000), 
                                                        X, 
                                                        y,
                                                        # Number of folds in cross-validation
                                                        cv=3,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 40 different sizes of the training set
                                                        train_sizes=[36, 54, 72])
    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # # Draw bands
    # plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    # plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

def test_all_chords_mlp(niter, max_iter, train_file, test_file):
    from sklearn.metrics import accuracy_score
    corretudes = []
    csv_name = "teste_mlp_niter={}_epocas={}.csv"
    real = []
    preditos = []
    for i in range(niter):
        clf = MLPClassifier(max_iter=max_iter)
        names = ["Freq1", "Freq2", "Freq3", "Freq4", "Freq5", "Freq6", "Freq7", "Freq8", "Freq9", "Classe"]
        df = pd.read_csv('NewDatasets/acordes_maiores.csv')
        df_train = pd.read_csv('NewDatasets/acordes_treino.csv')
        df_test = pd.read_csv('NewDatasets/acordes_teste.csv')
        X, y = df[df.columns[:5]], df[df.columns[-1]]
        x_train, y_train = df_train[df_train.columns[:5]], df_train[df_train.columns[-1]]
        x_test, y_test = df_test[df_test.columns[:5]], df_test[df_test.columns[-1]]
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        print([coef.shape for coef in clf.coefs_])
        corretudes.append(clf.score(x_test, y_test) * 100)
    filename = 'teste_MLP_niter={}_epocas={}.png'.format(niter, max_iter)
    title = r'Execução do MLP, $N={}, $epocas={}$'.format(niter, max_iter)
    save_hist(filename, title, corretudes)
    print(corretudes)

def test_two_chords_mlp(niter, max_iter, chord1_file, chord2_file):
    pass

def variar_parametros_mlp(param, param_range):
    names = ["Freq1", "Freq2", "Freq3", "Freq4", "Freq5", "Freq6", "Freq7", "Freq8", "Freq9", "Classe"]
    df = pd.read_csv('NewDatasets/acordes_maiores.csv')
    df_train = pd.read_csv('NewDatasets/acordes_treino.csv')
    df_test = pd.read_csv('NewDatasets/acordes_teste.csv')
    X, y = df[df.columns[:5]], df[df.columns[-1]]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    plot_validation_curve(x_scaled, y, param, param_range)


def main():
    # niter = 1
    # teste_mlp(niter)

    #variar_parametros_mlp("hidden_layer_sizes", [x for x in range(10, 110, 10)]) 
    # variar_parametros_mlp("max_iter", [x for x in range(1000, 2200, 200)])
    # variar_parametros_mlp("max_iter", [200, 500, 1000, 2000, 8000, 16000, 32000, 64000])
    # names = ["Freq1", "Freq2", "Freq3", "Freq4", "Freq5", "Freq6", "Freq7", "Freq8", "Freq9", "Classe"]
    #test_all_chords_mlp(20, 2500, 'NewDatasets/acordes_treino.csv', 'NewDatasets/acordes_teste.csv')
    df_train = pd.read_csv('NewDatasets/acordes_maiores.csv')
    x_train, y_train = df_train[df_train.columns[:5]], df_train[df_train.columns[-1]]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    example_plot_learning_curve(x_train_scaled, y_train)
    #print(df_train)

    #print(y_train)
    # grid, best_params = get_params(x_train_scaled, y_train)
    # print(best_params)

if __name__ == '__main__':
    main()

