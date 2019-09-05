import numpy as np
import random
import pandas as pd
import sklearn.preprocessing as preprocessing
import classificador

path_file1 = 'DataSets/widgets/training-widget.tab'
file1 = pd.read_csv(path_file1, sep='\t')

path_file2 = 'DataSets/widgets/test-widget.tab'
file2 = pd.read_csv(path_file2, sep='\t')

frames = [file1, file2]
dataset = pd.concat(frames)

labels = ['Dropdown', 'Suggestionbox', 'Other'] # labels do dataset
labelencoder = preprocessing.LabelEncoder() # codificador do sklearn
labelencoder = labelencoder.fit(labels) # codifica as labels em valores de 0 até n-1 / 1 - other / 2 
bc_labelencoded = labelencoder.transform(dataset.widgetClass) # faz a transformação da codificação para os indexes no dataset
# 0 - dropdown
# 1 - other
# 2 - suggestionbox

dataset['target'] = bc_labelencoded # salva na variável target o valor das classes codificadas
dataset = dataset.drop(columns="widgetClass")

labels = ['none', 'mouseover', 'keyup', 'click'] # labels do dataset
labelencoder = preprocessing.LabelEncoder() # codificador do sklearn
labelencoder = labelencoder.fit(labels) # codifica as labels em valores de 0 até n-1 / 1 - other / 2 
bc_labelencoded = labelencoder.transform(dataset.event) # faz a transformação da codificação para os indexes no dataset

dataset.event = bc_labelencoded

decisionTree = classificador.DecisionTree()
decisionTree.classificar(dataset)

knn = classificador.KNN()
knn.classificar(dataset)

svm = classificador.SVM()
svm.classificar(dataset)

randomForest = classificador.RandomForest()
randomForest.classificar(dataset)

logisticRegression = classificador.LR()
logisticRegression.classificar(dataset)