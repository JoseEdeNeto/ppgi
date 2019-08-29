import random
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV, train_test_split
from matplotlib import pyplot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class Classificador():
       def __init__ (self, seed):
              self._seed = seed

       def classificar(self, dataset):
              X = dataset[:,:-1] # seleciona todos os atributos menos o ultimo
              y = dataset['target'] # seleciona o "rótulo" que será classificado
              random.seed(self._seed) # seed

              ## StratifiedKfold separa os folds preservando a porcentagem de amostras para cada classe do target ##
              # n_splits = número de folds
              # random_state = seed
              kfold = StratifiedKFold(n_splits = 10, random_state = self._seed)
              modelo = self.get_classificador(X, y) #? TODO

              ## cv = método de cross validation
              ## scoring = métricas para avaliação do modelo
              ## f1_macro = calcula f1 separado por classes, mas sem estabelecer pesos
              scores = cross_validate(modelo, X, y, cv = kfold, scoring = ['accuracy', 'f1_macro', 'precision', 'recall'])
              print("Accuracy: %s on average and %s SD" %
                     (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
              print("Precision: %s on average and %s SD" %
                     (scores['test_precision'].mean(), scores['test_precision'].std()))
              print("Recall: %s on average and %s SD" %
                     (scores['test_recall'].mean(), scores['test_recall'].std()))
              print("F-measure: %s on average and %s SD" %
                     (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()))

              X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
              modelo.fit(X_train, y_train)

              dataset['%s_scores' % self.classificador] = scores
              return dataset

class DecisionTreeClassificador(Classificador):
    def __init__ (self, seed = 42):
        self._seed = seed