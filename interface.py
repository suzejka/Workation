import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from airlines import airlines
from hotels import hotels
from places_covered import places_covered
from sightseeing_places import sightseeing_places
import pickle
from joblib import dump, load
import warnings
warnings.filterwarnings("ignore")

st.image("https://www.greenpearls.com/wp-content/uploads/2018/09/puri-dajuma-eco-resort-bali.png")
col1, col2 = st.columns([1,2])
st.title("Holiday price prediction")
st.header("Enter your holiday details")

package_type = st.selectbox('Package Type', ('Standard', 'Deluxe', 'Premium','Luxury', 'Budget'))

package = package_type

places_covered = st.multiselect('Places Covered', places_covered)

itinerary = {
    place: st.slider('Itinerary {0}'.format(place), 1, 4)
    for place in places_covered
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
    encoder = load('encoder_{0}.joblib'.format(col.replace(' ', '_')))
    encoders[col] = encoder

for col in ['Package Type', 'Start City']:
    encoder = LabelEncoder()
    labels = np.load('encoder_{0}.npy'.format(col.replace(' ', '_')), allow_pickle=True)
    encoder.fit(labels)
    encoders[col] = encoder


if submit_button:

    package = encoders['Package Type'].transform([package])

    places_covered = encoders['Places Covered'].transform([places_covered])

    itinerary = encoders['Itinerary'].transform([itinerary])

    travel_date = pd.to_datetime(travel_date, errors='coerce')
    travel_date = travel_date.timestamp()    

    hotel_details = encoders['Hotel Details'].transform([hotel_details])

    start_city = encoders['Start City'].transform([start_city])

    airline = encoders['Airline'].transform([airline])

    sightseeing_places_covered = encoders['Sightseeing Places Covered'].transform([sightseeing_places_covered])

    userData = {'Package Type': package,
                'Places Covered': places_covered,
                'Itinerary': itinerary,
                'Travel Date': travel_date,
                'Hotel Details': hotel_details,
                'Start City': start_city,
                'Airline': airline,
                'Flight Stops': flight_stops,
                'Meals': meals,
                'Sightseeing Places Covered': sightseeing_places_covered}


    userDataFrame = pd.DataFrame.from_dict([userData])

    st.write(userDataFrame)

    model = pickle.load(open('models\\best_model.sv', 'rb'))

    prediction = model.predict(userDataFrame)

    # st.write(f"The price of the holiday is PLN {prediction[0]:.2f}")
