import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('Iris.csv')
setosa_data = df[df['Species'] == 'Iris-setosa']
versicolor_data = df[df['Species'] == 'Iris-versicolor']
virginica_data = df[df['Species'] == 'Iris-virginica']
setosa_data.head(5) // describe()
numerical_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
setosa_variability = setosa_data[numerical_cols].var()
versicolor_variability = versicolor_data[numerical_cols].var()
virginica_variability = virginica_data[numerical_cols].var()
print(setosa_variability)
setosa_std = setosa_data[numerical_cols].std()
versicolor_std = versicolor_data[numerical_cols].std()
virginica_std = virginica_data[numerical_cols].std()
print(setosa_std)
sns.scatterplot(data=df, x='SepalLengthCm', y='SepalWidthCm', hue='Species')
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Iris Dataset')
plt.show()