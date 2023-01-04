import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from pycaret.regression import *

LABEL_ENCODER = None

def remove_null(data):
    data.dropna(inplace=True)
    return data

def convert_dates_to_one_format(data):
    data['Travel Date'] = pd.to_datetime(data['Travel Date'], errors='coerce')
    data['Travel Date'] = data['Travel Date'].dt.strftime('%Y-%m-%d')
    return data

def replace_space_with_underscore(name):
    return name.replace(' ', '_')

def create_label_encoding(data):
    global LABEL_ENCODER
    cat_cols = data.select_dtypes(include=['object']).columns
    LABEL_ENCODER = LabelEncoder()
    for col in cat_cols:
        data[col] = LABEL_ENCODER.fit_transform(data[col])
        np.save('classes_{0}.npy'.format(replace_space_with_underscore(col)), LABEL_ENCODER.classes_, allow_pickle=True)

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

def find_best_models():
    rgs1 = setup(data = train, target = "Per Person Price")
    best_regression_models = compare_models()
    return best_regression_models

def create_best_model():
    best_model = create_model('rf') #random forest chosen from find_best_model()
    return best_model

train = pd.read_csv('dataset\Train.csv')
validation = pd.read_csv('dataset\Test.csv')

data_preprocessing(train)
data_preprocessing(validation)

best = find_best_models()
results = pull()
print(results) #printing table of best models

print(best) #printing best model found
#evaluate_model(best) #only works in notebook

save_model(best, 'random_forest_model') #saves best model as pickle file

