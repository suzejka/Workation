import pandas as pd

df = pd.read_csv('classes_Places_Covered.csv')

places_covered = df['Places Covered'].tolist()