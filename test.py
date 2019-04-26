import pandas as pd 
import numpy as np

csv_file = pd.read_csv('data.csv')
for i, row in enumerate(csv_file):
    if i == 3:
        print(row)