import pandas as pd
import os
file_path = os.path.sep.join(['encoders', 'classes_Hotel_Details.csv'])
df = pd.read_csv(file_path)

hotels = df['Hotel Details'].tolist()