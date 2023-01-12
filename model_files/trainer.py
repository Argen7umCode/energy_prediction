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
                
    def __make_step(self, sequence, labels, is_train=True):
        self.optimizer.zero_grad()
        self.model.hidden_cell = (
            torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device),
            torch.zeros(1, 1, self.model.hidden_layer_size).to(self.device)
        )

        y_pred = self.model(sequence)

        self.single_loss = self.loss_function(y_pred, labels)
        if is_train:
            self.single_loss.backward()
            self.optimizer.step()
    
    def __make_mileage_according_data(self, data, is_train=True):
        epoch_loss = []
        for sequence, labels in data:
            sequence = sequence.view((-1, 1))
            labels = labels.view((-1, 1))

            self.__make_step(sequence, labels, is_train)
            loss = self.single_loss.item()
            epoch_loss.append(loss)

        mean_epoch_loss = torch.mean(torch.tensor(epoch_loss))
        self.loss_log['train' if is_train else 'test'].append(mean_epoch_loss)

        
    def control_loss(self, loss):
        if loss <= self.min_loss:
            self.min_loss = loss
            self.best_parameters = self.model.state_dict()
            self.current_count_decreasing = 0
        else:
            self.current_count_decreasing += 1
            if self.current_count_decreasing >= self.max_count_decreasing:
                return 'stop'

    def make_epoch(self):
        self.model.train()
        self.__make_mileage_according_data(self.train_dataset)
        mean_epoch_loss = self.loss_log['train'][-1]
        print(f'LOSS TRAIN {mean_epoch_loss:10.6f}')

        # TODO Сделать пробег циклом по test_dataset и сохранять статистику loss по каждой из эпох 
        # TODO Сделать раннюю остановку - проверку на уменьшение loss, если он не уменьшается 
        #      max_count_decreasing раз подряд, то обучение прекращается
        #      если уменьшается, то сохраняем loss и параметры модели в best_model_parameters
         
        self.model.eval()
        with torch.no_grad():
            self.__make_mileage_according_data(self.test_dataset, False)
            mean_epoch_loss = self.loss_log['test'][-1]
        print(f'LOSS TEST {mean_epoch_loss}')

        if not self.control_loss(mean_epoch_loss):
            return True

    def fit(self):
        for epoch in range(self.n_epochs):
            if self.make_epoch() == 'stop':
                break
    
    def get_best_model(self):
        self.model.load_state_dict(self.best_parameters)
        return self.model

