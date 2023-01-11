import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from min_max_scaler import MinMaxScaler
from dataset import EnergyChickenFactory

def get_sequences(data, window_lenght):
    sequences = []
    for i in range(len(data)-window_lenght):
        sequences.append([
            data[i:i+window_lenght].to(device),
            data[i+window_lenght:i+window_lenght+1].to(device)
        ])
    return sequences

def split_data(data, spliter_count):
    spliter_count = int(len(data) * spliter_count)
    return data[:spliter_count], data[spliter_count:]



df = pd.read_csv('data/processed.csv')
df.drop(columns=df.columns[0], inplace=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
scaler = MinMaxScaler([-1, 1])
print(device)

data = torch.FloatTensor(scaler.fit_transform(df.iloc[:, 0].to_numpy()))[:100]

window_lenght = 2
dataset = EnergyChickenFactory(data, window_lenght)

print(list(dataset))





