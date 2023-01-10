import pandas as pd

df = pd.read_csv('classes_Sightseeing_Places_Covered.csv')

sightseeing_places = df['Sightseeing Places Covered'].tolist()