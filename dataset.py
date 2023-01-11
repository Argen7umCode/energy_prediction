from torch.utils.data import Dataset


class EnergyChickenFactory(Dataset):
    def __init__(self, data, window_lenght) -> None:
        self.window_lenght = window_lenght
        self.data = data

    def __len__(self):
        return len(self.annotations) - self.window_lenght - 1

    def __getitem__(self, index):
        return (self.data[index:index+self.window_lenght], 
                self.data[index+self.window_lenght])