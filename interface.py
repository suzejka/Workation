import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
from airlines import airlines
from hotels import hotels
from places_covered import places_covered
from sightseeing_places import sightseeing_places
import pickle

with open('models/ridge_model.pkl', 'rb') as file:
    model = pickle.load(file)

encoders = {}
for col in ['Package Name', 'Package Type', 'Destination', 'Itinerary', 'Places Covered', 'Travel Date', 'Hotel Details', 'Start City', 'Airline', 'Sightseeing Places Covered', 'Cancellation Rules']:
    encoder = LabelEncoder()
    encoder.classes_ = np.load('classes_{0}.npy'.format(col.replace(' ', '_')), allow_pickle=True)
    encoders[col] = encoder

def predict_price(package_type, places_covered, itinerary, travel_date, hotel_details, start_city, airline, flight_stops, meals, sightseeing_places_covered):
    X = np.array([
        encoders['Package Type'].transform([package_type]),
        encoders['Places Covered'].transform([places_covered]),
        encoders['Itinerary'].transform([itinerary]),
        encoders['Travel Date'].transform([travel_date]),
        encoders['Hotel Details'].transform([hotel_details]),
        encoders['Start City'].transform([start_city]),
        encoders['Airline'].transform([airline]),
        flight_stops,
        meals,
        encoders['Sightseeing Places Covered'].transform([sightseeing_places_covered]),
    ]).T

    y_pred = model.predict(X)[0]
    return y_pred


st.image("https://www.greenpearls.com/wp-content/uploads/2018/09/puri-dajuma-eco-resort-bali.png")
col1, col2 = st.columns([1,2])
col1.title("Holiday price prediction")
col2.header("Enter your holiday details")

form = col2.form(key='my_form')

package_type = form.selectbox('Package Type', ('Standard', 'Deluxe', 'Premium','Luxury', 'Budget'))
places_covered = form.multiselect('Places Covered', places_covered)
itinerary = form.multiselect('Itinerary', (1,1,1,2,2,2,2,3,3,3,3,4,4,4,4))
travel_date = form.date_input(label='Travel Date')
hotel_details = form.multiselect('Hotel Details', hotels)
start_city = form.selectbox('Start City', ('Mumbai', 'New Delhi') )
airline = form.selectbox('Airline', airlines)
flight_stops = form.slider('Flight Stops', 0, 2)
meals = form.slider('Meals', 2, 5)
sightseeing_places_covered = form.selectbox('Sightseeing Places Covered', sightseeing_places)

submit_button = form.form_submit_button(label='Send')

if submit_button:
    itinerary_combined = []
    for place, night in zip(places_covered, itinerary):
        itinerary_combined.append(f"{night}N {place}")
    price = predict_price(package_type, places_covered, itinerary_combined, travel_date, hotel_details, start_city, airline, flight_stops, meals, sightseeing_places_covered)
    st.write("The price of the holiday is PLN", price)