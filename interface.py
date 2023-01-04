import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder

encoders = {}

st.markdown("Przewidywanie ceny wakacji")

st.markdown("Wprowadź dane dotyczące Twoich wakacji")

form = st.form(key='my_form')

package_name = form.text_input(label='Package Name')
package_type = form.text_input(label='Package Type')
destination = form.text_input(label='Destination')
itinerary = form.text_input(label='Itinerary')
places_covered = form.text_input(label='Places Covered')
travel_date = form.text_input(label='Travel Date')
hotel_details = form.text_input(label='Hotel Details')
start_city = form.text_input(label='Start City')
airline = form.text_input(label='Airline')
flight_stops = form.text_input(label='Flight Stops')
meals = form.text_input(label='Meals')
sightseeing_places_covered = form.text_input(label='Sightseeing Places Covered')
cancellation_rules = form.text_input(label='Cancellation Rules')

submit_button = form.form_submit_button(label='Wyślij')

for col in ['Package Name', 'Package Type', 'Destination', 'Itinerary', 'Places Covered', 'Travel Date', 'Hotel Details', 'Start City', 'Airline', 'Sightseeing Places Covered', 'Cancellation Rules']:
    encoder = LabelEncoder()
    encoder.classes_ = np.load('classes_{0}.npy'.format(col.replace(' ', '_')), allow_pickle=True)
    encoders[col] = encoder

if submit_button:
    st.write(encoders['Package Name'].transform([package_name]))
    st.write(encoders['Package Type'].transform([package_type]))
    st.write(encoders['Destination'].transform([destination]))
    st.write(encoders['Itinerary'].transform([itinerary]))
    st.write(encoders['Places Covered'].transform([places_covered]))
    st.write(encoders['Travel Date'].transform([travel_date]))
    st.write(encoders['Hotel Details'].transform([hotel_details]))
    st.write(encoders['Start City'].transform([start_city]))
    st.write(encoders['Airline'].transform([airline]))
    st.write()
    st.write()
    st.write(encoders['Sightseeing Places Covered'].transform([sightseeing_places_covered]))
    st.write(encoders['Cancellation Rules'].transform([cancellation_rules]))
    st.write("Cena wakacji to 1000 zł")
