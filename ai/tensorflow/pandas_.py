"""Pandas test

"""
import pandas as pd
import numpy as np

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
population = pd.Series([852469, 1015785, 485199])

print(city_names)
print(population)
print(population / 100)
print(np.log(population))
print(population.apply(lambda val: val > 1000000))

cities = pd.DataFrame({ 'City name': city_names, 'Population': population })
print(type(cities['City name']))
print(cities['City name'])
print(type(cities['City name'][1]))
print(cities['City name'][1])
print(type(cities[0:2]))
print(cities[0:2])
print(cities)

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities['Area square miles']
print(cities)

california_housing_dataframe = pd.read_csv("./data/california_housing_train.csv", sep=",")
print(california_housing_dataframe)
print(california_housing_dataframe.describe())
print(california_housing_dataframe.head())
california_housing_dataframe.hist('housing_median_age')
