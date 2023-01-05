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
    rgs1 = setup(data = train, target = "Per Person Price", silent=True)
    best_regression_models = compare_models()
    return best_regression_models

def create_best_model():
    best_model = create_model('rf') #random forest chosen from find_best_model()
    return best_model
    
### preprocessing 

def split_category(value):
    vals = []
    if '|' in value:
        vals = value.split('|')
    else:
        vals.append(value)
    return vals

def split_columns_with_multiple_values(data):
  data['Airline'] = data['Airline'].apply(split_category)
  data['Destination'] = data['Destination'].apply(split_category)
  data['Places Covered'] = data['Places Covered'].apply(split_category)
  data['Sightseeing Places Covered'] = data['Sightseeing Places Covered'].apply(split_category)
  return data

def one_hot_encoding_on_columns(data):
  dummy_type = pd.get_dummies(data['Package Type'], prefix='type')
  data.drop(columns=['Package Type'], inplace=True)
  data = pd.concat([data, dummy_type], axis=1)

  dummy_city = pd.get_dummies(data['Start City'], prefix='sc')
  data.drop(columns=['Start City'], inplace=True)
  data = pd.concat([data, dummy_city], axis=1)

  label_encoder2 = LabelEncoder().fit(data['Cancellation Rules'])
  data['Cancellation Rules'] = label_encoder2.transform(data['Cancellation Rules'])
  return data


def show_category(series):
    values = {}
    for val in series:
        for each in val:
            if each in values:
                values[each] += 1
            else:
                values[each] = 1
    return values

def make_feature_col(series, all_keys):
    feature_dict = {}
    for key in all_keys:
        feature_dict[key] = []

    for items in series:
        for key in all_keys:
            if key not in items:
                feature_dict[key].append(0)
            else:
                feature_dict[key].append(1)

    return pd.DataFrame(feature_dict)

def structuring_columns(data):
  change = lambda pc: [each+'_ae' for each in pc]
  A_all_keys = show_category(data['Airline']).keys()
  A_all_keys = change(A_all_keys)
  airline = make_feature_col(data['Airline'], A_all_keys)
  data = pd.concat([data, airline], axis=1)

  change = lambda pc: [each+'_ds' for each in pc]
  D_all_keys = show_category(data['Destination']).keys()
  D_all_keys = change(D_all_keys)
  destination = make_feature_col(data['Destination'], D_all_keys)
  data = pd.concat([data, destination], axis=1)

  change = lambda pc: [each+'_pc' for each in pc]
  PC_all_keys = show_category(data['Places Covered']).keys()
  PC_all_keys = change(PC_all_keys)
  p_covered = make_feature_col(data['Places Covered'], PC_all_keys)
  data = pd.concat([data, p_covered], axis=1)

  return data
  
  
  
def preprocess_data(data):
  data = split_columns_with_multiple_values(data)
  data = one_hot_encoding_on_columns(data)
  data = structuring_columns(data)
  
 ### splitting data 
 
def split_data(data):
  features = ['Flight Stops', 'Meals', 'Cancellation Rules']
  other_features = data.columns[13:].to_list()
  features.extend(other_features)
  target = ['Per Person Price']

  X = data[features]
  y = data[target]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

  print('Shape of train set:', X_train.shape)
  print('Shape of test set:', X_test.shape)

  return X_train, X_test, y_train, y_test

train = pd.read_csv('dataset\Train.csv')
validation = pd.read_csv('dataset\Test.csv')

#data_preprocessing(train)
#data_preprocessing(validation)

data = preprocess_data(train)
X_train, X_test, y_train, y_test = split_data(data)

best = find_best_models()
results = pull()
print(results) #printing table of best models

print(best) #printing best model found
#evaluate_model(best) #only works in notebook

save_model(best, 'random_forest_model') #saves best model as pickle file