import pandas as pd 
import numpy as np

csv_file = pd.read_csv('data.csv')
data = []
i = 0
print(csv_file.loc[3:6])