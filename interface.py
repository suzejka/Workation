import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import warnings
from pycaret.regression import *
from sklearn.preprocessing import LabelEncoder
from airlines import airlines
from hotels import hotels
from places_covered import places_covered
from sightseeing_places import sightseeing_places

warnings.filterwarnings("ignore")
import os
file_path = os.path.abspath("")

def set_one_if_value_in_columns(df, columns, value):
    for col in columns:
        if col == value:
            df[col] = 1
        else:
            if col not in df.columns:
                df[col] = 0

def read_file_and_get_df(column):
    df = pd.read_csv('encoders\\classes_{0}.csv'.format(column.replace(' ', '_')))
    list = df[column].tolist()
    return pd.DataFrame(columns=list)

st.image("https://www.greenpearls.com/wp-content/uploads/2018/09/puri-dajuma-eco-resort-bali.png")
col1, col2 = st.columns([1,2])
st.title("Holiday price prediction")
st.header("Enter your holiday details")

package_type = st.selectbox('Package Type', ('Standard', 'Deluxe', 'Premium','Luxury', 'Budget'))

package = package_type

places_covered_selected = st.multiselect('Places Covered', places_covered)

itinerary = {
    place: st.slider('Itinerary {0}'.format(place), 1, 4)
    for place in places_covered_selected
}

travel_date = st.date_input(label='Travel Date')

hotel_details = st.multiselect('Hotel Details', hotels)
hotel_details = [hotel.lower() for hotel in hotel_details]

start_city = st.selectbox('Start City', ('Mumbai', 'New Delhi') )

airline = st.selectbox('Airline', airlines)

flight_stops = st.slider('Flight Stops', 0, 2)

meals = st.slider('Meals', 2, 5)

sightseeing_places_covered = st.multiselect('Sightseeing Places Covered', sightseeing_places)

submit_button = st.button(label='Send')

encoders = {}

for col in ['Itinerary', 'Sightseeing Places Covered', 'Places Covered', 'Hotel Details',  'Airline']:
    encoder = load('encoders\\encoder_{0}.joblib'.format(col.replace(' ', '_')))
    encoders[col] = encoder

for col in ['Package Type', 'Start City']:
    encoder = LabelEncoder()
    labels = np.load('encoders\\encoder_{0}.npy'.format(col.replace(' ', '_')), allow_pickle=True)
    encoder.fit(labels)
    encoders[col] = encoder

if submit_button:

    package = encoders['Package Type'].transform([package])

    travel_date = pd.to_datetime(travel_date, errors='coerce')
    travel_date = travel_date.timestamp()

    start_city = encoders['Start City'].transform([start_city])

    userData = pd.DataFrame()
    userData['Package Type'] = package
    userData['Travel Date'] = travel_date
    userData['Start City'] = start_city
    userData['Flight Stops'] = flight_stops
    userData['Meals'] = meals

    places_covered_df = pd.DataFrame(columns=places_covered)
    itinerary_df = read_file_and_get_df('Itinerary')
    hotels = [hotel.lower() for hotel in hotels]
    hotel_details_df = pd.DataFrame(columns=hotels)
    airline_df = pd.DataFrame(columns=airlines)
    sightseeing_places_covered_df = pd.DataFrame(columns=sightseeing_places)
    
    for i in itinerary:
        set_one_if_value_in_columns(userData, itinerary_df.columns, i)

    for place in places_covered_selected:
        set_one_if_value_in_columns(userData, places_covered_df.columns, place)  

    for hotel in hotel_details:
        set_one_if_value_in_columns(userData, hotel_details_df.columns, hotel)
        
    set_one_if_value_in_columns(userData, airline_df.columns, airline)

    for place in sightseeing_places_covered:
        set_one_if_value_in_columns(userData, sightseeing_places_covered_df.columns, place)
            
    model = load_model('models/best_model_pipeline')

    prediction = model.predict(userData)

    st.write(f"The price of the holiday is PLN {prediction[0]:.2f}")
    