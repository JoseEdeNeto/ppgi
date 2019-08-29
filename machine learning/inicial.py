import numpy as np
import random
import pandas as pd
import sklearn.preprocessing as preprocessing
import classificador

path_file1 = 'C:/personalstuff/DataSets/DataSets/training-widget.tab'
file1 = pd.read_csv(path_file1, sep='\t')

path_file2 = 'C:/personalstuff/DataSets/DataSets/test-widget.tab'
file2 = pd.read_csv(path_file2, sep='\t')

frames = [file1, file2]
dataset = pd.concat(frames)

print(dataset.head(40))

labels = ['Dropdown', 'Suggestionbox', 'Other'] # labels do dataset
labelencoder = preprocessing.LabelEncoder() # codificador do sklearn
labelencoder = labelencoder.fit(labels) # codifica as labels em valores de 0 até n-1 / 1 - other / 2 
bc_labelencoded = labelencoder.transform(dataset.widgetClass) # faz a transformação da codificação para os indexes no dataset

dataset['target'] = bc_labelencoded # salva na variável target o valor das classes codificadas

print(dataset['target'].value_counts())
