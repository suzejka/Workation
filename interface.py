import streamlit as st
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from airlines import airlines
from hotels import hotels
from places_covered import places_covered
from sightseeing_places import sightseeing_places
import pickle

st.image("https://www.greenpearls.com/wp-content/uploads/2018/09/puri-dajuma-eco-resort-bali.png")
col1, col2 = st.columns([1,2])
col1.title("Holiday price prediction")
col2.header("Enter your holiday details")

package_type = st.selectbox('Package Type', ('Standard', 'Deluxe', 'Premium','Luxury', 'Budget'))

package = package_type

places_covered = st.multiselect('Places Covered', places_covered)

itinerary = {}

for place in places_covered:
    itinerary[place] = st.slider('Itinerary {0}'.format(place), 1,4)

travel_date = st.date_input(label='Travel Date')

hotel_details = st.multiselect('Hotel Details', hotels)
hotel_details = [hotel.lower() for hotel in hotel_details]

start_city = st.selectbox('Start City', ('Mumbai', 'New Delhi') )

airline = st.selectbox('Airline', airlines)

flight_stops = st.slider('Flight Stops', 0, 2)

meals = st.slider('Meals', 2, 5)

sightseeing_places_covered = st.selectbox('Sightseeing Places Covered', sightseeing_places)

submit_button = st.button(label='Send')

encoders = {}
for col in ['Itinerary', 'Sightseeing Places Covered', 'Places Covered', 'Hotel Details',  'Airline']:
    encoder = MultiLabelBinarizer()
    labels = np.load('encoder_{0}.npy'.format(col.replace(' ', '_')), allow_pickle=True)
    labels = np.expand_dims(labels, axis=1)
    encoder.fit(labels)
    encoders[col] = encoder
    
for col in ['Package Type', 'Start City']:
    encoder = OneHotEncoder()
    labels = np.load('encoder_{0}.npy'.format(col.replace(' ', '_')), allow_pickle=True)
    encoder.fit(labels)
    encoders[col] = encoder

if submit_button:
    st.write(encoders['Package Type'].transform([package]))
    st.write(encoders['Places Covered'].transform([places_covered]))
    st.write(encoders['Itinerary'].transform([itinerary]))
    st.write()
    st.write(encoders['Hotel Details'].transform([hotel_details]))
    # st.write(encoders['Start City'].transform([start_city]))
    st.write(encoders['Airline'].transform([airline]))
    st.write()
    st.write()
    st.write(encoders['Sightseeing Places Covered'].transform([sightseeing_places_covered]))

    #read pickle model
    model = pickle.load(open('models\\ridge_model.pkl', 'rb'))

    #predict

    model.predict()

    st.write("The price of the holiday is PLN 1,000")
