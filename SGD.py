import numpy as np 
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score 
x = np.array([[1],[2],[3],[4],[5]])
y = np.array([[12],[18],[25],[27],[35]])
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
y_scaled = scaler.fit_transform(y)
x_scaled,y_scaled
model = SGDRegressor(
    learning_rate="constant",
    eta0=0.1,
    max_iter=1000,
    tol=1e-3,
    random_state=42
)
model.fit(x_scaled, y)
y_pred = model.predict(x_scaled)
print("Model Parameters :")
print("slope (m):",model.coef_[0])
print("intercept (c):",model.intercept_)

print("\nEvaluation Metrics :")
print("MSE :", mean_squared_error(y,y_pred))
print("MAE :", mean_squared_error(y,y_pred))
print("RMSE :",np.sqrt(mean_squared_error(y,y_pred)))
print("R^2 :",r2_score(y,y_pred))
x_new = np.array([[6]])
x_new_scaled = scaler.transform(x_new)
print("\nPrediction for 6 study Hours:",model.predict(x_new_scaled)[0])
