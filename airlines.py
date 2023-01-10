import pandas as pd

df = pd.read_csv('encoders\\classes_Airline.csv')

airlines = df['Airline'].tolist()