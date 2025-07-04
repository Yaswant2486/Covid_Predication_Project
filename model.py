import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

def create_sequences(data, time_steps=7):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(len(data_scaled) - time_steps):
        X.append(data_scaled[i:i + time_steps])
        y.append(data_scaled[i + time_steps])
    return np.array(X), np.array(y), scaler

def build_and_train(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model