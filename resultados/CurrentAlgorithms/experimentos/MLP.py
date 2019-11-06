import pandas as pd

from run import write_csv, save_hist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix


def get_params():
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

def teste_mlp(niter):
    from sklearn.metrics import accuracy_score
    corretudes = []
    csv_name = "teste_mlp.csv"
    real = []
    preditos = []
    for i in range(niter):
        clf = MLPClassifier()
        names = ["Freq1", "Freq2", "Freq3", "Freq4", "Freq5", "Freq6", "Freq7", "Freq8", "Freq9", "Classe"]
        df_train = pd.read_csv('TmpDataSets/acordes_treino.csv', names=names)
        df_test = pd.read_csv('TmpDataSets/acordes_teste.csv', names=names)
        x_train, y_train = df_train[df_train.columns[:9]], df_train[df_train.columns[-1]]
        x_test, y_test = df_test[df_test.columns[:9]], df_test[df_test.columns[-1]]

        # clf.set_params(clf2.best_params_)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        print(accuracy_score(y_test, y_predict))
        print(clf.score(x_test, y_test))
        # corretudes.append(clf.score(x_test, y_test))
        
    print(corretudes)

teste_mlp(niter)