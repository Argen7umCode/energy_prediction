import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from utils.get_sequences import get_sequences
from utils.split_data import split_data
from sklearn.preprocessing import MinMaxScaler


# df = pd.read_csv('data/processed.csv')
# df.drop(columns=df.columns[0], inplace=True)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# scaler = MinMaxScaler([-1, 1])
# print(device)

# data = torch.FloatTensor(scaler.fit_transform(df.iloc[:, 0].to_numpy().reshape((-1, 1)))).to(device)

# print(data)
# window_lenght = 10
# sequences = get_sequences(data, window_lenght)

# train, test = split_data(sequences, 0.8)


# TODO поиграться с параметрами и архитектурой модели для получения лучшей метрики
class LSTM(nn.Module):
    def __init__(self, device, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.device = device
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size).to(self.device),
                            torch.zeros(1,1,self.hidden_layer_size).to(self.device))

    def forward(self, input_seq):
        input_seq.to(self.device)

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1).to(self.device), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# model = LSTM(device=device)
# model.to(device)
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# epochs = 150
# best_loss = np.inf
# best_params = model.parameters()
# n = 0

# for i in range(epochs):
#     for seq, labels in train[:100]:
#         optimizer.zero_grad()
#         model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
#                         torch.zeros(1, 1, model.hidden_layer_size).to(device))
        
#         y_pred = model(seq)

#         single_loss = loss_function(y_pred, labels)
#         single_loss.backward()
#         optimizer.step()

#     loss = single_loss.item()

#     if loss <= best_loss:
#         best_loss = loss
#     else:
#         n += 1
#         if n > 5:
#             model.parameters = best_params 

#     print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
