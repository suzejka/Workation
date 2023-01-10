import pandas as pd

df = pd.read_csv('encoders\\classes_Places_Covered.csv')

places_covered = df['Places Covered'].tolist()