from sklearn.datasets import load_iris, fetch_california_housing 
import pandas as pd

iris = load_iris(as_frame=True)
iris_df = iris.frame
print(iris_df.head())

housing = fetch_california_housing(as_frame=True)
housing_df = housing.frame
print(housing_df.describe())