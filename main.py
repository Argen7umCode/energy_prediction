# TODO посторить пайплайн работы модели 
#      перестроить весь файл
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from utils.get_sequences import get_sequences
from utils.split_data import split_data
from sklearn.preprocessing import StandardScaler
from data_files.data_module import EnergyChickenDataModule
from model_files.model import LSTM
from model_files.trainer import Trainer
from torch.optim import Adam
import matplotlib.pyplot as plt


RANDOM_SEED = 42
SPLITER_COUNT = 0.8
WINDOW_LENGHT = 4 * 7 * 24 
BATCH_SIZE = 32
MAX_COUNT_DECREASING = 10
LEARNING_RATE = 0.001
N_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv('data_files/processed.csv')
df.drop(columns=df.columns[0], inplace=True)
scaler = StandardScaler()
data = torch.FloatTensor(scaler.fit_transform(df.iloc[:, 0].to_numpy().reshape((-1, 1)))).to(DEVICE)[:]

sequences = get_sequences(data, WINDOW_LENGHT)
train, test = split_data(sequences, SPLITER_COUNT)

data_module = EnergyChickenDataModule(train, test)
data_module.setup()
model = LSTM(device=DEVICE)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

trainer = Trainer(model=model, 
                  optimizer=optimizer, 
                  loss_function=criterion,
                  n_epochs=N_EPOCHS,
                  device=DEVICE,
                  max_count_decreasing=MAX_COUNT_DECREASING)

train_dataloader = data_module.train_dataloader()
test_dataloader = data_module.test_dataloader()
trainer.upload_data(train_dataloader, test_dataloader)



trainer.fit()

stat = trainer.logs
train_loss = stat['train_loss']
test_loss = stat['test_loss']
min_train_loss = min(train_loss)
min_train_loss_epoch = train_loss.index(min_train_loss)
min_test_loss = min(test_loss)
min_test_loss_epoch = test_loss.index(min_test_loss)
print(f'train_loss {(1-min_train_loss**0.5)*100}\ntest_loss {(1-min_test_loss**0.5)*100} ')

plt.scatter([min_train_loss_epoch], [min_train_loss], label='Min train loss')
plt.scatter([min_test_loss_epoch], [min_test_loss], label='Min test loss')
plt.plot(train_loss, label='Train loss')
plt.plot(test_loss, label='Test loss')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


