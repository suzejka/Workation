import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import streamlit as st

def remove_null(data):
    data.dropna(inplace=True)
    return data

def convert_dates_to_one_format(data):
    data['Travel Date'] = pd.to_datetime(data['Travel Date'], errors='coerce')
    data['Travel Date'] = data['Travel Date'].dt.strftime('%Y-%m-%d')
    return data

def create_label_encoding(data):
    cat_cols = data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])

def change_int32_to_int64(data):
    for col in data.columns:
        if data[col].dtype == 'int64':
            data[col] = data[col].astype('int32')

def data_preprocessing(data):
    data.drop(['Uniq Id'], axis=1, inplace=True)
    remove_null(data)
    change_int32_to_int64(data)
    convert_dates_to_one_format(data)
    create_label_encoding(data)

def build_neural_network_for_regression():
    network = Sequential()
    network.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=13))
    network.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    network.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return network

def build_linear_regression(X_train, y_train):
    classifier = LinearRegression()
    classifier.fit(X_train, y_train)
    return classifier

train = pd.read_csv('dataset\Train.csv')
validation = pd.read_csv('dataset\Test.csv')

data_preprocessing(train)
data_preprocessing(validation)

X = train.iloc[:, 0:13].values
y = train.iloc[:, 13].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regression = build_linear_regression(X, y)

y_pred = regression.predict(X_test)

accuracy = regression.score(X_test, y_test)
print(accuracy)

st.markdown("Przewidywanie ceny wakacji")