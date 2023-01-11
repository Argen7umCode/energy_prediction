import torch


class Trainer:
    def __init__(self, model, optimizer, loss_function, n_epochs, device, max_count_decreasing) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        self.device = device
        self.best_parameters = self.model.parameters()
        self.min_loss = torch.inf
        self.current_count_decreasing = 0
        self.max_count_decreasing = max_count_decreasing
        self.loss_log = {'train' : [], 'test' : []}

    def upload_data(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
                
    def make_step(self, sequence, labels, train=True):
        self.optimizer.zero_grad()
        self.model.hidden_cell = (
            torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device),
            torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device)
        )
        y_pred = self.model(sequence)

        self.single_loss = self.loss_function(y_pred, labels)
        if train:
            self.single_loss.backward()
            self.optimizer.step()
    
    def make_epoch(self):
        self.model.train()
        for sequence, labels in self.train_dataset:
            self.make_train_step(sequence, labels)
        loss = self.single_loss.item()

        self.model.eval()
        with torch.no_grad():
            for sequence, labels in self.test_dataset:
                self.make_train_step(sequence, labels)
            loss = self.single_loss.item()
        
        if not self.control_loss(loss):
            return False
        
    def control_loss(self, loss):
        if loss <= self.min_loss:
            self.min_loss = loss
            self.best_parameters = self.model.parameters()
            self.current_count_decreasing = 0
        else:
            self.current_count_decreasing += 1
            if self.current_count_decreasing >= self.max_count_decreasing:
                return False

    def fit(self):
        for epoch in range(self.n_epochs):
            if not self.make_epoch():
                break
        
        

    def test(self):
        
        with torch.no_grad():
            for sequence, labels in self.test_dataset:
                self.make_train_step(sequence, labels)
            loss = self.single_loss.item()

            if not self.control_loss(loss):
                return False