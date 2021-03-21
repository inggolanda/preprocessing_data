# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 15:44:12 2021

@author: inggo
"""

# Mengimpor library yang diperlukan
import numpy as np
import pandas as pd
 
# Import data ke python dan memisahkan Variabel dependen dan independen
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
 
# Memproses data yang hilang (missing) mengisi nilai kosong dengan rata rata
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
 
# Encoding data kategori dan variabel independen
#enkoding = 2 kolom yang sifatnya kategori (categorical) yaitu kolom ‘Negara’ dan ‘Beli’ menadi format angka
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
#lencoder_X = LabelEncoder()                     
#X[:, 0] = lencoder_X.fit_transform(X[:, 0])
#Note :Sklearn Versi Lama Kategori telah berubah menjadi 0 = France , Germany = 1, Spain = 2 namun python menganggap 1 dan 2 lebih besar dari angka sebelumnya

#Dummy Variabel 
transformer = ColumnTransformer(
        #Negara dan 0 menyatakan kolom yang di inginkan
        [('Negara', OneHotEncoder(), [0])],
        remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)
 
# Encode variabel dependen(Beli)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)