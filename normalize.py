#charger donnees du csv
#data = pd.read_csv('crc_h_bit_glcm_.csv')
# Sélectionner les colonnes à normaliser
#columns_to_normalize = ['X', 'Y']
# Créer l'instance du MinMaxScaler
#
# Ajuster le scaler sur les données et normaliser les colonnes sélectionnées
#data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
# Afficher les données normalisées
#print(data)
# Enregistrer les données normalisées dans un nouveau fichier CSV
#data.to_csv('donnees_normalisees.csv', index=False)
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn .preprocessing import MinMaxScaler
filename = 'crc_h_bit_glcm_.csv'
#dataframe lit le contenu fichier
dataframe = read_csv(filename)
#passer les valeurs du dataframe dans un tableau
array = dataframe.values
X = array[:, 0:-1]
Y = array[:, -1]
#instance MinMaxScaler
scaler = MinMaxScaler()
#preparation des donnees en apprentissage(transformation)
Rescaled_X = scaler.fit_transform(X)
#combinaison du X transforme et du Y(changer la dimension dun tableau)
#réorganiser les données du tableau tout en conservant le même nombre total d'éléments.
#data = np.hstack(Rescaled_X,Y.reshape(-1,1)) 
data = np.hstack((Rescaled_X, Y.reshape(-1, 1)))
#creation new dataframe
data1 = pd.DataFrame(data)
data1.to_csv('crc_.csv',index=False,header=False)