# task3.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os


df = pd.read_csv(r"C:\Users\Betraaj\Downloads\Desktop\internship\elevate labs\TASK 3\Housing.csv")  


df = df.dropna()


print("Available columns:", df.columns.tolist())


X = df[['area']]                 
y = df['price']               
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")


print(f"\nIntercept: {model.intercept_}")
print(f"Coefficient(s): {model.coef_}")


if X.shape[1] == 1:
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Line')
    plt.xlabel(X.columns[0])
    plt.ylabel('Price')
    plt.title('Linear Regression - Actual vs Predicted')
    plt.legend()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/regression_plot.png")
    print("\nPlot saved to: plots/regression_plot.png")
    plt.show()

