import pandas as pd
import os
file_path = os.path.sep.join(['encoders', 'classes_Sightseeing_Places_Covered.csv'])
df = pd.read_csv(file_path)

sightseeing_places = df['Sightseeing Places Covered'].tolist()