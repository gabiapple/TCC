import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning
class Resultado():
    def __init__(self, y, predict_y):
        """
        y: Vetor numpy (np.array) em que, para cada instancia i, y[i] é a classe alvo da mesma
        predict_y: Vetor numpy (np.array) que cada elemento y[i] representa a predição da instancia i

        Tanto y quando predict_y devem assumir valores numéricos
        """
        self.y = y
        self.predict_y = predict_y
        self._mat_confusao = None
        self._precisao = None
        self._revocacao = None

    @property
    def mat_confusao(self):
        """
        Retorna a matriz de confusão.
        """
        #caso a matriz de confusao já esteja calculada, retorna-la
        if self._mat_confusao  is not None:
            return self._mat_confusao

        #instancia a matriz de confusao como uma matriz de zeros
        #A matriz de confusão terá o tamanho como o máximo entre os valores de self.y e self.predict_y
        max_class_val = max([self.y.max(),self.predict_y.max()])
        self._mat_confusao = np.zeros((max_class_val+1,max_class_val+1))



        #incrementa os valores da matriz baseada nas listas self.y e self.predict_y
        for i,classe_real in enumerate(self.y):
            self._mat_confusao[classe_real][self.predict_y[i]] += 1

        # print("Predict y: "+str(self.predict_y))
        # print("y: "+str(self.y))
        # print("Matriz de confusao final :"+str(self._mat_confusao))
        return self._mat_confusao


    @property
    def acuracia(self):
        #quantidade de elementos previstos corretamente
        num_previstos_corretamente = 0
        for classe in range(len(self.mat_confusao)):
            num_previstos_corretamente += self.mat_confusao[classe][classe]
        return num_previstos_corretamente/len(self.y)

    @property
    def precisao(self):
        """
        Precisão por classe
        """
        if self._precisao is not None:
            return self._precisao

        #inicialize com um vetor de zero usando np.zeros
        self._precisao = np.zeros(len(self.mat_confusao))

        for classe in range(len(self.mat_confusao)):
            soma = 0
            for classe_prevista in range(len(self.mat_confusao)):
                soma += self.mat_confusao[classe_prevista][classe]
           
            if soma:
                self._precisao[classe] = self.mat_confusao[classe][classe]/soma
            else:
                self._precisao[classe] = 0
                warnings.warn("undefinedMetricWarning", UndefinedMetricWarning )
        return self._precisao
    @property
    def revocacao(self):
        if self._revocacao is not None:
            return self._revocacao

        self._revocacao = np.zeros(len(self.mat_confusao))
        for classe in range(len(self.mat_confusao)):
            soma = 0
            for classe_prevista in range(len(self.mat_confusao)):
                soma += self.mat_confusao[classe][classe_prevista]

            #armazene no veotr a precisão para a classe, não deixe que ocorra divisão por zero
            #caso ocorra, faça um warning
            if soma:
                self.revocacao[classe] = self.mat_confusao[classe][classe]/soma
            else:
                self.revocacao[classe] = 0
                warnings.warn("undefinedMetricWarning", UndefinedMetricWarning )
        return self._revocacao

    @property
    def f1_por_classe(self):
        """
        retorna um vetor em que, para cada classe, retorna o seu f1
        """

        #f1 é o veor de f1 que deve ser retornado. Altere o "None" com o tamanho correto do vetor
        self._f1_por_classe = np.zeros(len(self.precisao))

        for classe in range(len(self.mat_confusao)):
            if(self.precisao[classe] or self.revocacao[classe]):
                num_f1 = 2 * self.precisao[classe] * self.revocacao[classe]
                den_f1 = self.precisao[classe] + self.revocacao[classe]
                self._f1_por_classe[classe] = num_f1/den_f1
            else:
               self._f1_por_classe[classe] = 0
        return self._f1_por_classe

    @property
    def macro_f1(self):
        """
        Calcula o macro f1
        """
        return np.average(self.f1_por_classe)
