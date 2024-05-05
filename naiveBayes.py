import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
X=iris.drop(['Species'], axis=1)
Y=iris['Species']
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
Y_train
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
Y_Pred = gnb.predict(X_test)
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score, confusion_matrix
cm = confusion_matrix(Y_test, Y_Pred)
print(cm)
accuracy = accuracy_score(Y_test,Y_Pred).round(2)
print(accuracy)
precision =precision_score(Y_test, Y_Pred,average='micro').round(2) #avg=weighted
print(precision)
recall =  recall_score(Y_test, Y_Pred,average='micro').round(2)
print(recall)
f1 = f1_score(Y_test,Y_Pred,average='micro')
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb.classes_)
disp.plot()