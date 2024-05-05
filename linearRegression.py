import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("Datasets/Banglore Housing Prices.csv")
df.head(5) df.describe()
df.isnull().sum()
df["bath"] = df["bath"].apply(lambda x: np.nan if pd.isnull(x) else x).isnull()
df["bath"].fillna(df["bath"].mean() , inplace=True)
df.dropna(inplace=True)
df["size"] = df["size"].apply(lambda x: int(x.split()[0]))
df["size"] = df["size"].astype("int")
df['total_sqft'].describe
def convert_sqft(value):
    try:
        if '-' in value:
            start, end = map(float, value.split('-'))
            return (start+end) / 2
        else:
            return float(value)
    except ValueError:
        return float('nan')
df['total_sqft'] = [convert_sqft(value) for value in df["total_sqft"]]
df.isnull().sum()
df.dropna(inplace=True)
df['Price_per_sqft'] = df['price'] / df['total_sqft']
df.head(5)
df.total_sqft.max()
plt.boxplot(df.total_sqft)
new = df[df["total_sqft"] < 100000000]
plt.boxplot(new.total_sqft)
new["total_sqft"].max()
plt.boxplot(new["size"])
new = new[new["size"] < 11]
plt.boxplot(new["size"])
"""
def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
df['Price_per_sqft'] = remove_outliers(df['Price_per_sqft'])
sns.boxplot(x=df['Price_per_sqft']) plt.show() """
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Define features and target variable
X = new[['size', 'total_sqft','bath','price_per_sqft']]
y = new['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Accuracy (R-squared): {r2}')