import pandas as pd

df = pd.read_csv('hotel_details.csv')

hotels = df['Hotel Details'].tolist()