import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#other basic import dataset , describe , null values
df['Embarked'].unique()
df['Embarked'] = df['Embarked'].fillna('S', inplace=True)
df['Age'] = df['Age'].fillna(df.Age.mean().round(2), inplace=True)
df['Cabin_New'] = df['Cabin'].fillna(0)
df['Cabin_New'] = df['Cabin'].notnull().astype(int)
df['survived'].value_counts()
#pie plot
plt.figure(figsize=(8,6))
plt.pie(survived, labels=['Not Survived', 'Survived']), autopct='%1.1f%%', colors=palette_color)
plt.title('Survival Status')
plt.show()
#countplot
sns.set(style='whitegrid')
plt.figure(figsize=(8,6))
sns.countplot(x='Survived', hue='Survived', data=df, palette=['black', 'cyan'])
plt.xlabel('Survived')
plt.ylabel('Count')
#Pclass
df['Pclass'].value_counts()
df.groupby('Pclass')['Survived'].value_counts()
sns.countplot(data=df,x='Pclass',hue='Survived')
plt.title('Survival Count By Pclass')
#By SEX
sns.countplot(data=df, x=Pclass, hue='Sex')
#Fare
plt.figure(figsize(10,8))
sns.histplot(data=df, x='fare', bins=20, color='blue')
plt.title("Distribution of Ticket Prices")
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
#BOxplot
plt.figure(figsize(10,8))
sns.boxplot(data=df, x='Sex', y='Age', hue='Survived')
plt.title("BOx Plot according to Gender")
#Violin Plot
plt.figure(figsize(10,8))
sns.violinplot(data=df, x='Sex', y='Age', hue='Survived', palette=['lightcoral', 'cyan'])
plt.title("BOx Plot according to Gender")
#IRIS
sns.histplot(data=df, x='sepal_width', hue='Species') # same for all features
sns.boxplot(data=df)
sns.boxplot(data=df,x='species',y=sepal_width) #same for all other