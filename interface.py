import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
from airlines import airlines
from hotels import hotels
from places_covered import places_covered
from sightseeing_places import sightseeing_places

st.image("https://www.greenpearls.com/wp-content/uploads/2018/09/puri-dajuma-eco-resort-bali.png")
col1, col2 = st.columns([1,2])
col1.title("Holiday price prediction")
col2.header("Enter your holiday details")

package_type = st.selectbox('Package Type', ('Standard', 'Deluxe', 'Premium','Luxury', 'Budget'))
places_covered = st.multiselect('Places Covered', places_covered)

itinerary = {}

for place in places_covered:
    itinerary[place] = st.slider('Itinerary {0}'.format(place), 1,4)

travel_date = st.date_input(label='Travel Date')
hotel_details = st.multiselect('Hotel Details', hotels)
start_city = st.selectbox('Start City', ('Mumbai', 'New Delhi') )
airline = st.selectbox('Airline', airlines)
flight_stops = st.slider('Flight Stops', 0, 2)
meals = st.slider('Meals', 2, 5)
sightseeing_places_covered = st.selectbox('Sightseeing Places Covered', sightseeing_places)

submit_button = st.button(label='Send')

encoders = {}
for col in ['Package Name', 'Package Type', 'Destination', 'Itinerary', 'Places Covered', 'Travel Date', 'Hotel Details', 'Start City', 'Airline', 'Sightseeing Places Covered', 'Cancellation Rules']:
    encoder = LabelEncoder()
    encoder.classes_ = np.load('classes_{0}.npy'.format(col.replace(' ', '_')), allow_pickle=True)
    encoders[col] = encoder

if submit_button:
    st.write(encoders['Package Type'].transform([package_type]))
    st.write(encoders['Places Covered'].transform([places_covered]))
    st.write(encoders['Itinerary'].transform([itinerary]))
    st.write(encoders['Travel Date'].transform([travel_date]))
    st.write(encoders['Hotel Details'].transform([hotel_details]))
    st.write(encoders['Start City'].transform([start_city]))
    st.write(encoders['Airline'].transform([airline]))
    st.write()
    st.write()
    st.write(encoders['Sightseeing Places Covered'].transform([sightseeing_places_covered]))
    st.write("The price of the holiday is PLN 1,000")
