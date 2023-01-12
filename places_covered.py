import pandas as pd
import os
file_path = os.path.sep.join(['encoders', 'classes_Places_Covered.csv'])
df = pd.read_csv(file_path)

places_covered = df['Places Covered'].tolist()