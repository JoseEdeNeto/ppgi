# https://docs.python.org/3/library/random.html
import random

# funções e operações para cálculos numéricos / https://docs.scipy.org/doc/numpy/reference/
import numpy as np
print("numpy version: {}". format(np.__version__))

# manipulação e processamento de dados / https://pandas.pydata.org/pandas-docs/stable/
import pandas as pd
print("pandas version: {}". format(pd.__version__))

# pré processamento de dados / https://scikit-learn.org/stable/modules/preprocessing.html
import sklearn.preprocessing as preprocessing

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Classificadores implementados em classificador.py
import classificador

path_file1 = 'DataSets/widgets/5-test-widget-com-url.tab'
file1 = pd.read_csv(path_file1, sep='\t')

path_file2 = 'DataSets/widgets/5-training-widget-com-url.tab'
file2 = pd.read_csv(path_file2, sep='\t')

frames = [file1, file2]
dataset = pd.concat(frames)

dataset = dataset[dataset.widgetClass != 'Discard']

dataset.loc[(dataset['widgetClass'] == 'Dropdown'), 'widgetClass'] = 0
dataset.loc[(dataset['widgetClass'] == 'Other'), 'widgetClass'] = 1
dataset.loc[(dataset['widgetClass'] == 'Suggestionbox'), 'widgetClass'] = 2

dataset.rename(columns={'widgetClass': 'target'}, inplace=True)


dataset.loc[(dataset['event'] == 'mouseover'), 'event'] = 0
dataset.loc[(dataset['event'] == 'click'), 'event'] = 1
dataset.loc[(dataset['event'] == 'none'), 'event'] = 2
dataset.loc[(dataset['event'] == 'keyup'), 'event'] = 3

decisionTree = classificador.DecisionTree()
decisionTree.classificar(dataset, "GroupKfold")

knn = classificador.KNN()
knn.classificar(dataset, "GroupKfold")

svm = classificador.SVM()
svm.classificar(dataset, "GroupKfold")

randomForest = classificador.RandomForest()
randomForest.classificar(dataset, "GroupKfold")

logisticRegression = classificador.LR()
logisticRegression.classificar(dataset, "GroupKfold")