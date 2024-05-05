import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('Social_Network_Ads.csv')
df.head(5)
df.Gender.unique()
df["IsMale"] = df.Gender
df["IsMale"] = df.Gender.apply(lambda x: 1 if x=="Male" else 0)
df.drop(["UserID","Gender"], axis=1, inplace=True)
df.head(5)
y=df['Purchased']
x = df[['Age', 'EstimatedSalary','IsMale']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
x_train
from sklearn.preprocessing import StandardScaler
std=StandardScaler()
x_test=std.fit_transform(x_test)
x_test
x_train=std.fit_transform(x_train)
x_train
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(xtrain, ytrain)
y_pred=classifier.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(ytest, y_pred)
sns.heatmap(cm, annot=True, cmap='coolwarm')
plt.title("Confusion Matrix")
plt.show()
accuracy = accuracy_score(ytest, y_pred)
error_rate = 1 - accuracy
precision = precision_score(ytest, y_pred)
recall = recall_score(ytest, y_pred)
f1 = f1_score(ytest, y_pred)