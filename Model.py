'''
Project - MLOps_POC
Salary Prediction
'''

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

location = "POC-data.xlsx"
df = pd.read_excel(location)

X = df.drop("Salary",axis = 1)
y = df["Salary"]

model = LinearRegression()
model.fit(X,y)

print(model.predict(np.array([10,10]).reshape(1,-1)))
