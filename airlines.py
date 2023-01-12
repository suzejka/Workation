import pandas as pd
import os

file_path = os.path.sep.join(['encoders', 'classes_Airline.csv'])
df = pd.read_csv(file_path)

airlines = df['Airline'].tolist()