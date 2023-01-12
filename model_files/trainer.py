import torch

# TODO оптимизировать как-то класс и его конструктор, чтоб он не был таким загроможденными 

class Trainer:
    def __init__(self, model, optimizer, loss_function, n_epochs, device, max_count_decreasing) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.n_epochs = n_epochs
        self.device = device
        self.best_parameters = self.model.state_dict()
        self.min_loss = torch.inf
        self.current_count_decreasing = 0
        self.max_count_decreasing = max_count_decreasing
        self.loss_log = {'train' : [], 'test' : []}
        self.model.to(self.device)

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
        # TODO Сделать пробег циклом по train_dataset и сохранять статистику loss по каждой из эпох 
        self.model.train()
        epoch_loss = []
        for sequence, labels in self.train_dataset:
            sequence = sequence.view((-1, 1))
            labels = labels.view((-1, 1))

            self.make_step(sequence, labels)
            loss = self.single_loss.item()
            epoch_loss.append(loss)
        mean_epoch_loss = torch.mean(torch.tensor(epoch_loss))
        self.loss_log['train'].append(mean_epoch_loss)
        
        print(f'LOSS TRAIN {mean_epoch_loss:10.6f}')

        # TODO Сделать пробег циклом по test_dataset и сохранять статистику loss по каждой из эпох 
        # TODO Сделать раннюю остановку - проверку на уменьшение loss, если он не уменьшается 
        #      max_count_decreasing раз подряд, то обучение прекращается
        #      если уменьшается, то сохраняем loss и параметры модели в best_model_parameters
         
        # self.model.eval()
        # with torch.no_grad():
        #     for sequence, labels in self.test_dataset:
        #         sequence = sequence.view((-1, 1))
        #         labels = labels.view((-1, 1))
        #         self.make_step(sequence, labels)
        #     loss = self.single_loss.item()
        # print(f'LOSS TEST {loss}')

        # if not self.control_loss(loss):
        #     return False
        
    def control_loss(self, loss):
        if loss <= self.min_loss:
            self.min_loss = loss
            self.best_parameters = self.model.state_dict()
            self.current_count_decreasing = 0
        else:
            self.current_count_decreasing += 1
            if self.current_count_decreasing >= self.max_count_decreasing:
                return False

    def fit(self):
        for epoch in range(self.n_epochs):
            if self.make_epoch():
                break
    
    def get_best_model(self):
        self.model.load_state_dict(self.best_parameters)
        return self.model

    def test(self):
        
        with torch.no_grad():
            for sequence, labels in self.test_dataset:
                self.make_train_step(sequence, labels)
            loss = self.single_loss.item()

            if not self.control_loss(loss):
                return False