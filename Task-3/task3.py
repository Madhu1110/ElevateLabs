
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("Housing.csv") 

# Select input and output 
X = df[["area"]]       
y = df["price"]       

# Split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model using MAE, MSE, and R² Score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print(f"Mean Absolute Error (MAE): ₹{mae:,.2f}")
print(f"Mean Squared Error (MSE): ₹{mse:,.2f}")
print(f"R² Score: {r2:.2f}")

# Visualize the regression line
plt.scatter(X_test, y_test, color='black', label="Actual Prices")
plt.plot(X_test, y_pred, color='green', linewidth=2, label="Regression Line")
plt.xlabel("Area")
plt.ylabel("Price ")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.show()
