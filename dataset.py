import dill
import os
from torch.utils.data import DataLoader

class Dataset():
    def __init__(
        self,
        data_dir,
        n_clients,
        load_train = True,
        load_test = True
    ):
        # load train data
        self.train_data = []
        if load_train:
            for i in range(n_clients):
                with open(os.path.join(data_dir, f"Client{i+1}.pkl"), "rb") as f:
                    self.train_data.append(dill.load(f))

        # load test data
        if load_test:
            with open(os.path.join(data_dir, "Test.pkl"), "rb") as f:
                self.test_data = dill.load(f)

    def get_test_dataloader(self, batch_size):
        return DataLoader(
            self.test_data, 
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
    
    def get_train_dataloaders(self, batch_size):
        train_loaders = []
        for data in self.train_data:
            train_loaders.append(DataLoader(
                data,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            ))
        return train_loaders