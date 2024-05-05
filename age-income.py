import pandas as pd
import numpy as np
df = pd.read_excel("Age-Income-Dataset.xlsx")
def calculate_mean(column):
  sum = 0
  count = 0
  for i in df[column]:
    sum += i
    count += 1
  mean = sum / count
  print(mean)
def calculate_median(column):
  sorted_data = sorted(df[column])
  count = len(df[column])
  if(count%2==0):
    #even
    median = (sorted_data[(count//2)-1] + sorted_data[(count//2)]) / 2
  else:
    median = sorted_data[(count//2)-1]
  print(median)
def find_min(column):
  min = df[column][0]
  for i in df[column]:
    if i < min:
        min = i
  print(min)
def find_max(column):
  max = df[column][0]
  for i in df[column]:
    if i > max:
        max = i
  print(max)
def std_dev(column):
  length = len(column)
  summation = 0
  total_sum = 0
  for value in column:
    total_sum += value
  mean = total_sum / length
  for i in column:
    summation += (i-mean)*(i-mean)
  std_deviation = (summation / length ) ** 0.5
  print(std_deviation)

df["Income"].mean()
df["Income"].median()
df["Income"].max()
df["Income"].min()
df["Income"].std()

df.groupby('Age')['Income'].mean()
df.groupby('Age')['Income'].median()
df.groupby('Age')['Income'].std()
df.groupby('Age')['Income'].min()
df.groupby('Age')['Income'].max()