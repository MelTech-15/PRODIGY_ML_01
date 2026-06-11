import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("House Price Prediction - Linear Regression\n")

# Load the raw CSV file into a pandas DataFrame
print("Loading Data from CSV")
df = pd.read_csv('train.csv')

# Printing the total rows and columns to verify import status
# print("Full Dataset Shape:", df.shape)

# Selecting the columns needed
print("Selecting the required features for modeling")
selected_columns = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']

data_df = df[selected_columns].rename(columns={
    'GrLivArea': 'SquareFootage',
    'BedroomAbvGr': 'Bedrooms',
    'FullBath': 'Bathrooms'
})

# Data Preview
# print(data_df.head())

X = data_df[['SquareFootage','Bedrooms','Bathrooms']]
Y = data_df['SalePrice']

print("Data Splitted for Testing and Training")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X,Y)

print("Training the model")

intercept = model.intercept_
coefficients = model.coef_

print("\nResults:")

print(f"Intercept: {intercept:.2f}")
for name,val in zip(data_df.columns,coefficients):
    print(f"{name}: {val:.2f}")

Y_predicted = model.predict(X_test)

# Metrics
RMSE = np.sqrt(mean_squared_error(Y_test, Y_predicted))
Rsq = r2_score(Y_test,Y_predicted)

print("\n--- Model Evaluation ---")
print(f"RMSE: {RMSE:.2f}")
print(f"R squared: {Rsq:.2f}")

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(Y_test,Y_predicted, color='blue', alpha=0.2, label='Predicted House Rates')
plt.plot([Y_test.min(),Y_test.max()],[Y_test.min(),Y_test.max()], color='red', lw=2, label='Line of Best Fit')

plt.legend()

plt.show()