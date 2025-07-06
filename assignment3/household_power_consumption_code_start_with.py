# %%
import numpy as np
import pandas as pd
import os

# %%
print("当前工作目录：", os.getcwd())

# %%
# load data
df = pd.read_csv(
    'c:/Users/30358/Desktop/aiSummerCamp2025-master/day3/assignment/data/household_power_consumption.txt',
    sep=";",
    encoding="latin-1",
    na_values="?",
    skiprows=0,
    nrows=10
)
print(df)

# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis = 1, inplace = True)
# handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']

# %%
# data normalization
from sklearn.preprocessing import MinMaxScaler

feature_cols = [col for col in train.columns if col != 'datetime' and col != 'Global_active_power']
target_col = 'Global_active_power'

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

train_X = scaler_X.fit_transform(train[feature_cols])
train_y = scaler_y.fit_transform(train[[target_col]])
test_X = scaler_X.transform(test[feature_cols])
test_y = scaler_y.transform(test[[target_col]])

# %%
# split X and y (reshape for LSTM: [samples, time_steps, features])
def create_sequences(X, y, seq_length=24):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(xs), np.array(ys)

seq_length = 24  # 24 hours as one sequence
X_train_seq, y_train_seq = create_sequences(train_X, train_y, seq_length)
X_test_seq, y_test_seq = create_sequences(test_X, test_y, seq_length)

# %%
# create dataloaders
import torch
from torch.utils.data import TensorDataset, DataLoader

batch_size = 64
X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_seq, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# build a LSTM model
import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # use the last output
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRegressor(input_size=X_train_seq.shape[2]).to(device)

# %%
# train the model
import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}")

# %%
# evaluate the model on the test set
model.eval()
preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        preds.append(output.cpu().numpy())
preds = np.concatenate(preds, axis=0)
preds_inv = scaler_y.inverse_transform(preds)
y_test_inv = scaler_y.inverse_transform(y_test_seq)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test_inv, preds_inv)
print(f"Test MSE: {mse:.4f}")

# %%
# plotting the predictions against the ground truth
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(y_test_inv[:500], label='True')
plt.plot(preds_inv[:500], label='Predicted')
plt.legend()
plt.title("LSTM Prediction vs Ground Truth")
plt.xlabel("Time Step")
plt.ylabel("Global Active Power")
plt.show()
