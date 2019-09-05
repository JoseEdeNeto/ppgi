import numpy as np
import random
import pandas as pd
import sklearn.preprocessing as preprocessing
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
              X = dataset.loc[:, dataset.columns != 'target'] ## seleciona todos os atributos menos o target
              y = dataset['target'] ## seleciona o "rótulo" que será classificado

              random.seed(self._seed) ## gera seed

              ## StratifiedKfold separa os folds preservando a porcentagem de amostras para cada classe do target ##
              # n_splits = número de folds
              # random_state = seed
              kfold = StratifiedKFold(n_splits = 10, random_state = self._seed)
              modelo = self.get_classificador(X, y)

              ## cv = método de cross validation
              ## scoring = métricas para avaliação do modelo
              ## f1_macro = calcula f1 separado por classes, mas sem estabelecer pesos
              scores = cross_validate(modelo, X, y, cv = kfold, scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'])
              print("Accuracy: %s on average and %s SD" %
                     (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
              print("Precision: %s on average and %s SD" %
                     (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()))
              print("Recall: %s on average and %s SD" %
                     (scores['test_recall_macro'].mean(), scores['test_recall_macro'].std()))
              print("F-measure: %s on average and %s SD" %
                     (scores['test_f1_macro'].mean(), scores['test_f1_macro'].std()))

              X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
              modelo.fit(X_train, y_train)

              return dataset

class DecisionTree(Classificador):
       def __init__ (self, seed = 42):
              Classificador.__init__(self, seed)
              self.tipo = 'decision_tree'

       def get_classificador(self, X, y):
              print('-- DECISION TREE --')
              model = DecisionTreeClassifier(random_state=self._seed)
              return model

class KNN(Classificador):
       def __init__ (self, seed = 42):
              Classificador.__init__(self, seed)
              self.tipo = 'decision_tree'

       def get_classificador(self, X, y):
              print('-- KNN --')
              model = KNeighborsClassifier()
              return model

class SVM(Classificador):
       def __init__ (self, seed = 42):
              Classificador.__init__(self, seed)
              self.tipo = 'decision_tree'

       def get_classificador(self, X, y):
              print('-- SVM --')
              model = SVC(random_state=self._seed)
              return model

class RandomForest(Classificador):
       def __init__ (self, seed = 42):
              Classificador.__init__(self, seed)
              self.tipo = 'decision_tree'

       def get_classificador(self, X, y):
              print('-- RANDOM FOREST --')
              model = RandomForestClassifier(random_state=self._seed)
              return model

class LR(Classificador):
       def __init__ (self, seed = 42):
              Classificador.__init__(self, seed)
              self.tipo = 'decision_tree'

       def get_classificador(self, X, y):
              print('-- LOGISTIC REGRESSION --')
              model = LogisticRegression(random_state=self._seed)
              return model