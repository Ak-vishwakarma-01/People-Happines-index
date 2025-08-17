# People Happiness Index

This project analyzes the relationship between GDP per capita and life satisfaction using Python and machine learning models.

## Data

The dataset `lifesat.csv` contains country-level data with columns:
- `Country`
- `GDP per capita (USD)`
- `Life satisfaction`

## Analysis

We use two types of regression models to predict life satisfaction from GDP per capita:

### Linear Regression

A model that assumes a linear relationship between GDP and life satisfaction. It fits a best-fit line minimizing the square errors.

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

text

### K-Nearest Neighbors Regression (KNN)

A non-parametric model that predicts life satisfaction based on averaging the values of the nearest neighbor countries in terms of GDP.

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

text

## Visualization

We create scatter plots to visualize the data:

lifesat.plot(kind='scatter', x='GDP per capita (USD)', y='Life satisfaction', grid=True)
plt.axis([23500,[62500])
plt.show()

text

The axis limits focus on the GDP and satisfaction ranges in the dataset.

## Prediction Example

Predict life satisfaction for Cyprus (GDP per capita 37,655.2 USD):

X_new = [[37655.2]]
print(model.predict(X_new))

text

## Installation

Make sure to install required packages with:

pip install scikit-learn matplotlib pandas numpy

text

---

This README covers the core project setup and usage.