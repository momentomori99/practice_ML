import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore") #its cuz boston housing dataset is unethical, and therefore it gives yuou warnings because of this.

#Loading the data:
from sklearn.datasets import load_boston
boston_dataset = load_boston()

#boston_dataset.keys()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()



boston['MEDV'] = boston_dataset.target

#boston.isnull().sum()


#for the visualization of the data, we can use seaborn:
#sns.set(rc={'figure.figsize':(11.7,8.27)})

# plot a histogram showing the distribution of the target values (but just for the MEDV)
#sns.distplot(boston['MEDV'], bins=30)
#plt.show()

#Looking at the correlation matrix
correlation_matrix = boston.corr().round(2)
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
#sns.heatmap(data=correlation_matrix, annot=True)
#From the correlation visualization we can clearly see that MEDV is strongly correlated to LSTAT and RM. Alså RAD with TAX.


# plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']
#
# for i, col in enumerate(features):
#     plt.subplot(1, len(features) , i+1)
#     x = boston[col]
#     y = target
#     plt.scatter(x, y, marker='o')
#     plt.title(col)
#     plt.xlabel(col)
#     plt.ylabel('MEDV')

#Since MEDV correlates well with LSTAT and RM, we make a new datafram with only these
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
#np.c_ here is to joing two arrays basically.
Y = boston['MEDV']

#Now we split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

#And now we simply just use linear regression functionality from scikit-learn
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)
y_train_predict = lin_model.predict(X_train)

#See how well it did with the training set:
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)
print(f"rmse: {rmse}")
print(f"r2: {r2}")

#Now move on to the test part of the data
y_test_predict = lin_model.predict(X_test)
# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)
print(f"rmse: {rmse}")
print(f"r2: {r2}")

plt.scatter(Y_test, y_test_predict)
plt.show()
