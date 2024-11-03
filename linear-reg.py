# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('boston.csv')

data_cleaned = data.drop(columns=['Unnamed: 0'])

X = data_cleaned.drop(columns=['PRICE'])  # Features
y = data_cleaned['PRICE'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print("Actual prices: ", y_test.head().values)
print("Predicted prices: ", y_pred[:5])

