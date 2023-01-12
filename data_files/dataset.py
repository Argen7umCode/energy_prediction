from torch.utils.data import Dataset
import torch


class EnergyChickenFactoryDataset(Dataset):
    def __init__(self, sequences) -> None:
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, label = self.sequences[index]

        return (torch.Tensor(sequence), torch.tensor(label).float())