class MinMaxScaler:
    def __init__(self, feature_range) -> None:
        self.feature_range = feature_range

    def fit(self, data):
        self.min = data.min()
        self.max = data.max()
    
    def transform(self, data):
        min = self.feature_range[0]
        max = self.feature_range[1]
        return (data - self.min) / (self.max - self.min) * (max - min) + min

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
