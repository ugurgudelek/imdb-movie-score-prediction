import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_boston

boston = load_boston()
boston.keys()


boston.data.shape


print(boston.feature_names)


print(boston.DESCR)


bos = pd.DataFrame(boston.data)
bos.head()


bos.columns = boston.feature_names
bos.head()


boston.target[:5]

bos['PRICE'] = boston.target
bos.head()


from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE', axis = 1)



# This creates a LinearRegression objects
lm = LinearRegression()


X_train = X[:-50]
X_test = X[-50:]
y_train = bos.PRICE[:-50]
y_test = bos.PRICE[-50:]


lm.fit(X_train, y_train)



print("intercept : \n", lm.intercept_)
print("coef : \n", lm.coef_)




d = pd.DataFrame(list(zip(X.columns, lm.coef_)), columns= ['features', 'estimatedCoefs'])
print(d)



mse = np.mean((y_test - lm.predict(X_test))**2)
print(mse)


plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, c='b', s=20, alpha = 0.5)
plt.scatter(lm.predict(X_test), lm.predict(X_test) - y_test, c='g', s=20)
plt.hlines(y=0, xmin=0, xmax = 50)
plt.title("Residual plot")
plt.ylabel("Residuals")
plt.rcParams["figure.figsize"] = (10,10)
plt.show()




