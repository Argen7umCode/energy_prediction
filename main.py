# TODO посторить пайплайн работы модели 
#      перестроить весь файл
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from utils.get_sequences import get_sequences
from utils.split_data import split_data
from sklearn.preprocessing import MinMaxScaler
from data_files.data_module import EnergyChickenDataModule
from model_files.model import LSTM
from model_files.trainer import Trainer
from torch.optim import AdamW


RANDOM_SEED = 42
SPLITER_COUNT = 0.8
WINDOW_LENGHT = 100
BATCH_SIZE = 32
MAX_COUNT_DECREASING = 5
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('data_files/processed.csv')
df.drop(columns=df.columns[0], inplace=True)
scaler = MinMaxScaler([-1, 1])
data = torch.FloatTensor(scaler.fit_transform(df.iloc[:, 0].to_numpy().reshape((-1, 1)))).to(DEVICE)
sequences = get_sequences(data, WINDOW_LENGHT)
train, test = split_data(sequences, SPLITER_COUNT)

data_module = EnergyChickenDataModule(train, test)
data_module.setup()
model = LSTM(device=DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

trainer = Trainer(model=model, 
                  optimizer=optimizer, 
                  loss_function=criterion,
                  n_epochs=WINDOW_LENGHT,
                  device=DEVICE,
                  max_count_decreasing=MAX_COUNT_DECREASING)

train_dataloader = data_module.train_dataloader()
test_dataloader = data_module.test_dataloader()
trainer.upload_data(train_dataloader, test_dataloader)

trainer.fit()

# for i in train_dataloader:
#     x = i[1].squeeze(dim=2)
#     print(x.shape)
#     print(x)
#     break





