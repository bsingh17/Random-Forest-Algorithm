import numpy as np
import pandas as pd

dataset=pd.read_csv('petrol_consumption.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
x=scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=20,random_state=0)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn import metrics

print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_predict))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_predict))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))