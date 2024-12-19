import numpy as np

class EarlyStopping():
    def __init__(self, delta=0.0001, patience=10):
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_model = None
        self.delta = delta
        self.patience = patience
        self.counter = 0
    
    def update(self, model, epoch, loss):
        early_stop = False
        if loss < self.best_loss - self.delta:
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter == self.patience:
                early_stop = True
                print(f"EarlyStopping: Stopped training at epoch {epoch}.")
        return early_stop