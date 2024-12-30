from atr import run, load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
# from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import os
# import numpy as np
# from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import random
import joblib
from tqdm import tqdm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.options.mode.copy_on_write = True


def find_first_threshold(dataframe, start_index, up_threshold, down_threshold):
    for i in range(start_index, len(dataframe)):
        if dataframe.iloc[i]['High'] >= up_threshold and dataframe.iloc[i]['Low'] <= down_threshold:
            return -1
        if dataframe.iloc[i]['High'] >= up_threshold:
            return 1
        if dataframe.iloc[i]['Low'] <= down_threshold:
            return 0
    return -1


def make_dataset(dataframe, window_size=400, atr_limit=1):
    x = []
    y = []
    for i in tqdm(range(len(dataframe) - window_size)):
        atr_up_threshold = dataframe.iloc[i + window_size - 1]['Close'] + atr_limit * \
                           dataframe.iloc[i + window_size - 1]['ATR']
        atr_down_threshold = dataframe.iloc[i + window_size - 1]['Close'] - atr_limit * \
                             dataframe.iloc[i + window_size - 1]['ATR']
        label = find_first_threshold(dataframe, i + window_size, atr_up_threshold, atr_down_threshold)
        if label != -1:
            x.append(dataframe.iloc[i:i + window_size].loc[:, ['Open', 'High', 'Low', 'Close']].to_numpy().tolist())
            y.append(label)
    return torch.tensor(x), torch.tensor(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h_0, c_0))
        # out = self.fc(self.norm(out[:, -1, :]))
        out = self.fc(out[:, -1, :])
        # out = self.fc2(out)
        # out = self.norm(out)
        return self.sigmoid(out)


def train(x, y, train_flag=True):
    input_size = 4
    hidden_size = 16
    num_layers = 3 # 2
    output_size = 1
    batch_size = 64
    learning_rate = 0.0001
    num_epochs = 100
    dropout = 0.5
    clip_value = 5.0
    wd = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMModel(input_size, hidden_size, num_layers, output_size, device, dropout).to(device)

    train_loss_plot = []
    val_loss_plot = []

    x_continuous = x[:, :, :]

    scaler = MinMaxScaler()
    num_samples, sequence_len, num_continuous_features = x_continuous.shape

    # Normalaize for each sample.
    for i in range(num_samples):
        x_continuous[i] = torch.tensor(scaler.fit_transform(x_continuous[i]), dtype=torch.float64)
    x_continuous_normalized = x_continuous

    # Normalize for all data.
    # x_continuous_reshaped = x_continuous.reshape(-1, num_continuous_features)
    # x_continuous_normalized = scaler.fit_transform(x_continuous_reshaped)
    # joblib.dump(scaler, f'{BASE_PATH}scaler.gz')

    # NO NORMALIZE
    # x_continuous_normalized = x_continuous_reshaped


    x_continuous_normalized = torch.tensor(x_continuous_normalized.reshape(num_samples, sequence_len, 
                                                                           num_continuous_features))

    x_preprocessed = torch.tensor(x_continuous_normalized, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    ### UNDER SAMPLING
    # Y = Y.squeeze()
    # class_0_indices = (Y == 0).nonzero(as_tuple=True)[0]
    # class_1_indices = (Y == 1).nonzero(as_tuple=True)[0]
    # min_class_samples = min(len(class_0_indices), len(class_1_indices))
    # undersampled_class_0_indices = class_0_indices[torch.randperm(len(class_0_indices))[:min_class_samples]]
    # undersampled_class_1_indices = class_1_indices[torch.randperm(len(class_1_indices))[:min_class_samples]]
    # balanced_indices = torch.cat([undersampled_class_0_indices, undersampled_class_1_indices])
    # balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]
    # X_balanced = X[balanced_indices]
    # Y_balanced = Y[balanced_indices]
    # Y_balanced = Y_balanced.view(-1, 1)
    # Y_balanced = Y_balanced.float()
    # X_balanced = balanced_data[:, :-1]
    # Y_balanced = balanced_data[:, -1:]
    ### UNDER SAMPLING

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)

    dataset = TensorDataset(x_preprocessed, y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))

    # train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
    #                                                         generator=torch.Generator().manual_seed(8))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    best_val_loss = float('inf')
    best_model_path = 'best_lstm_model.pth'
    if train_flag:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                outputs = outputs.reshape([-1, 1])
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()

                running_loss += loss.item()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs).squeeze()
                    outputs = outputs.reshape([-1, 1])
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_loss_plot.append(avg_train_loss)
            avg_val_loss = val_loss / len(val_loader)
            val_loss_plot.append(avg_val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at epoch {epoch + 1} with validation loss {best_val_loss:.4f}")
        plt.plot(train_loss_plot, label='train_loss')
        plt.plot(val_loss_plot, label='val_loss')
        plt.legend()
        plt.savefig(f'loss_plot.png')
    # Test the model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    predict_label = []
    true_label = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            outputs = outputs.reshape([-1, 1])
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # predicted = (outputs >= 0.5).float()
            ##### MAJORRRRRRR
            # outputs = torch.rand(outputs.size()).to(device)
            predicted = outputs.round()
            # print(f'Outputs: {outputs}')
            predict_label.extend(outputs.view(-1).tolist())
            true_label.extend(labels.view(-1).tolist())
            # print(f'Predicted: {predicted}')
            # print(f'True Label: {labels}')
            # print((predicted == labels).sum(), (predicted == labels).sum().item())
            correct_predictions += (predicted == labels).sum()
            total_samples += labels.size(0)

    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct_predictions / total_samples

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return predict_label, true_label


def max_drawdown(numbers):
    max_length = 0
    current_length = 1 

    for i in range(1, len(numbers)):
        if numbers[i] < numbers[i - 1]:
            current_length += 1
        else:
            max_length = max(max_length, current_length)
            current_length = 1

    max_length = max(max_length, current_length) 
    return max_length
    
    
def plot_balance(predict_label, true_label, thresh):
    balance = [100]
    profit = 100
    for p, t in zip(predict_label, true_label):
        if p < thresh:
            if t == 0:
                profit += 1
            else:
                profit -= 1
            balance.append(profit)
        elif p >= 1 - thresh:
            if t == 1:
                profit += 1
            else:
                profit -= 1
            balance.append(profit)
    print(f'Max Drawdown: {max_drawdown(balance)}')
    fig = px.line(x=list(range(len(balance))), y=balance, markers=True)
    fig.show()
    return balance


def main():
    BASE_PATH = 'XAUUSD/'
    path = f'{BASE_PATH}train.csv'
    x_path = BASE_PATH + 'x.pickle'
    y_path = BASE_PATH + 'y.pickle'
    if os.path.exists(x_path) and os.path.exists(y_path):
        with open(x_path, 'rb') as file:
            X = pickle.load(file)
        with open(y_path, 'rb') as file:
            Y = pickle.load(file)
    else:
        df = load_dataset(path)
        df_result = run(df)
        X, Y = make_dataset(df_result, 400, 1)
        with open(x_path, 'wb') as file:
            pickle.dump(X, file)
        with open(y_path, 'wb') as file:
            pickle.dump(Y, file)
    Y = Y.reshape(-1, 1)
    print(f'X Size: {X.size()}')
    print(f'Y Size: {Y.size()}')
    print(f'Count of 0: {(Y == 0).sum()}')
    print(f'Count of 1: {(Y == 1).sum()}')
    pl, tl = train(X, Y, True)
    plot_balance(pl, tl, 0.5)


if __name__ == '__main__':
    main()
