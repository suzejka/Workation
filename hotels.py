import pandas as pd

df = pd.read_csv('encoders\\classes_Hotel_Details.csv')

hotels = df['Hotel Details'].tolist()