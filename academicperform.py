import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    df.drop(outliers.index, inplace=True)
def box_plot(dataframe, column):
    sns.set(style="whitegrid")
    plt.figure(figsize=(8,6))
    sns.boxplot(x=dataframe[column])
    plt.title('Boxplot of {}'.format(column))
    plt.xlabel('Column')
    plt.ylabel('Values')
    plt.show()
def mean_inplace(df, column):
    mean_value = df[column].mean()
    df[column].fillna(mean_value, inplace=True)
academic_performance.info()
#Missing value percentage
for column in academic_performance.columns:
    miss_per = (academic_performance[column].isnull().sum() / len(academic_performance))
    print(f"Percentage of missing values in {column}: {miss_per: .2f}%")
#Dealing with missing gender
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(academic_performance[['Gender']])
academic_performance['Gender'] = imputer.transform(academic_performance[['Gender']])
#dealing with missing placement
academic_performance.dropna(subset=['Placement'], inplace=True)
#Dealing with Honor
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(academic_performance[['Honor_Opted_OR_Not']])
academic_performance['Honor_Opted_OR_Not'] = imputer.transform(academic_performance[['Honor_Opted_OR_Not']])
#Dealing with Education_type
imputer = SimpleImputer(strategy='most_frequent')
imputer.fit(academic_performance[['Education_Type']])
academic_performance['Education_Type'] = imputer.transform(academic_performance[['Education_Type']])
#Dealing with Academic program missing
academic_performance.dropna(subset=['Academic_program'], inplace=True)
acadamic_performace.isnull().sum()
#Remove Outliers
remove_outliers(academic_performance, 'Course 1 Marks') #upto course5marks
remove_outliers(academic_performance, 'Percentile')
#Remove Null values
mean_inplace(academic_performance, 'Course 1 marks') #upto course5marks
performance_categorical = academic_performance.select_dtypes(exclude=[np.number])
performance_categorical = performance_categorical.drop('StudentID', axis=1)
performance_categorical
performance_categorical.EducationType.value_counts()
performance_categorical.placement.replace({"Yes": 1, "No": -1}, inplace=True)
performance_categorical.gender.replace({"M": 1, "F": -1}, inplace=True)
performance_categorical.honor.replace({"Yes": 1, "No": -1}, inplace=True)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
performance_categorical['Overall_grade']=label_encoder.fit_transform(performance_categorical['Overall_grade'])
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(df[['academicprogram', 'educationtype']])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['academicprogram', 'educationtype']))
print(encoded_df)
