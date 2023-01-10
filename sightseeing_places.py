import pandas as pd

df = pd.read_csv('encoders\\classes_Sightseeing_Places_Covered.csv')

sightseeing_places = df['Sightseeing Places Covered'].tolist()