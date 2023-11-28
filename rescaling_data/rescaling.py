import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import numpy as np
import pandas as pd
from IPython.display import display

np.random.seed(100)

rows = 10
cols = 5

X = np.random.randn(rows, cols)



X_p = pd.DataFrame(X)

print(X_p.mean())
print(X_p.std())

X_p = X_p - X_p.mean()
print(X_p)


#  This option does not include the standard deviation
scaler = StandardScaler(with_std=False)
scaler.fit(X)
Xscaled = scaler.transform(X)
display(X_p-Xscaled)
