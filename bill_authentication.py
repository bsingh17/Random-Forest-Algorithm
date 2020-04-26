import numpy as np
import pandas as pd

dataset=pd.read_csv('bill_authentication.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(x)
x=scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20,random_state=0)
model.fit(x_train,y_train)
y_predict=model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_predict)
print(confusion)

print(model.score(x_test,y_test))