import csv
import numpy as np

csv_file = csv.reader(open('data.csv', 'r'))
data = []
i = 0
for stu in csv_file:
    data.append(stu)
    i = i + 1
    print('load ', i, 'images')

print(np.shape(data))