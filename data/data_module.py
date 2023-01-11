# import torch
# import pandas as pd
from dataset import EnergyChickenFactory
from torch.utils.data import DataLoader


# df = pd.read_csv('data/processed.csv')
# df.drop(columns=df.columns[0], inplace=True)
# data = torch.FloatTensor(scaler.fit_transform(df.iloc[:, 0].to_numpy()))

# TODO дописать этот класс, написать что-то!!!
class DataModule:
    def __init__(self, train_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self):
        self.train_dataset = EnergyChickenFactory(self.train_sequences)
        self.test_dataset = EnergyChickenFactory(self.test_sequences)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size = 1, 
            shuffle=False,
            num_workers=1
        )

        
    # def __init__(self, data, window_lenght, input_count=1, output_count=1, spliter_count=0.8) -> None:
    #     self.data = data
    #     self.window_lenght = window_lenght
    #     self.input_count = input_count
    #     self.output_count = output_count
    #     self.spliter_count = int(len(self.data) * spliter_count)
    
    # def get_sequences(self):
    #     sequences = []
    #     for i in range(len(self.data)-self.window_lenght-self.output_count):
    #         sequences.append([
    #             data[i:i+self.window_lenght],
    #             data[i+self.window_lenght:i+self.window_lenght+self.output_count]
    #         ])
    #     return sequences

    # def split_data(self):
    #     return {'sequence' : self.data[:self.spliter_count], 
    #             'label': self.data[self.spliter_count:]}
            
    # def setup(self):
