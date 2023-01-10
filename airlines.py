import pandas as pd

df = pd.read_csv('airlines.csv')

airlines = df['Airline'].tolist()