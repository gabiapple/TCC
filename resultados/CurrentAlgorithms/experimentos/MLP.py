import pandas as pd
from sklearn.neural_network import MLPClassifier

niter=1
for i in range(niter):
    clf = MLPClassifier()
    df_train = pd.read_csv('TmpDataSets/acordes_treino.csv', header=None)
    df_test = pd.read_csv('TmpDataSets/acordes_teste.csv', header=None)
    x_train, y_train = df_train[df_train.columns[:9]], df_train[df_train.columns[-1]]
    x_test, y_test = df_test[df_test.columns[:9]], df_test[df_test.columns[-1]]
    
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


    print("Best parameters set found on development set:")
    print(clf2.best_params_)

    clf.set_params(clf2.best_params_)
    clf.fit(x_train, y_train)
    print("Score MLP:")
    print(clf.score(x_test, y_test))
