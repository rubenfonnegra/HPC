
import pandas as pd

data = pd.read_csv("healthcare-dataset-stroke-data.csv")

print ("=".center(50, '='))
print ("Dataset information".center(50, '='))
print ("=".center(50, '='))

print (data.info())

print ("=".center(50, '='))
print ("Descriptive analysis".center(50, '='))
print ("=".center(50, '='))

print (data.describe())
