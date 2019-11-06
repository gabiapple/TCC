import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_learning_curve import plot_learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from run import write_csv, save_hist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix


def get_params(x_train, y_train):
    from sklearn.model_selection import GridSearchCV
    param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
            ],
            'max_iter' : [200, 500, 1000]
        }
       ]
    clf2 = GridSearchCV(MLPClassifier(), param_grid, cv=3,
                        scoring='accuracy')
    clf2.fit(x_train, y_train)
    return clf2.best_params_

def plot_validation_curve(X,y):
    # Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    from sklearn.model_selection import validation_curve
    param_range = np.logspace(-6, -2, 4)
    
    train_scores, test_scores = validation_curve(MLPClassifier(), X, y, "learning_rate_init",
                                               param_range,
                                               cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    title = "MPL"
    plt.title("Validation Curve with MPL")
    plt.xlabel(r"$learning_rate$")
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2,
                    color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

    plot_learning_curve(MLPClassifier(), title, X, y)
    plt.show()

def example_plot_learning_curve(X, y):
    # Reference: https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/
    train_sizes, train_scores, test_scores = learning_curve(MLPClassifier(), 
                                                        X, 
                                                        y,
                                                        # Number of folds in cross-validation
                                                        cv=4,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.01, 1.0, 50))
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

def teste_mlp(niter):
    from sklearn.metrics import accuracy_score
    max_it = 8000
    corretudes = []
    csv_name = "teste_mlp.csv"
    real = []
    preditos = []
    for i in range(niter):
        clf = MLPClassifier(max_iter=max_it)
        names = ["Freq1", "Freq2", "Freq3", "Freq4", "Freq5", "Freq6", "Freq7", "Freq8", "Freq9", "Classe"]
        df = pd.read_csv('TmpDataSets/acordes_ordenados.csv', names=names)
        df_train = pd.read_csv('TmpDataSets/acordes_treino.csv', names=names)
        df_test = pd.read_csv('TmpDataSets/acordes_teste.csv', names=names)
        X, y = df[df.columns[:9]], df[df.columns[-1]]
        x_train, y_train = df_train[df_train.columns[:9]], df_train[df_train.columns[-1]]
        x_test, y_test = df_test[df_test.columns[:9]], df_test[df_test.columns[-1]]
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        corretudes.append(clf.score(x_test, y_test) * 100)
    filename = 'teste_MLP_niter={}_epocas={}.png'.format(niter, max_it)
    title = r'Histograma da execução do MLP, $N={}, $epocas={}$'.format(niter, max_it)
    save_hist(filename, title, corretudes)
    print(corretudes)

niter = 20
teste_mlp(niter)