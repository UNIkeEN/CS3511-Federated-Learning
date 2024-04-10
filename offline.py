import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
from tqdm import tqdm
import models
from dataset import Dataset
from utils import *

class OfflinePipeline():
    def __init__(self, cfg):
        self.device = cfg.device
        self.client_ckp_dir = cfg.client_ckp_dir
        self.global_ckp_dir = cfg.global_ckp_dir
        check_directory(self.client_ckp_dir)
        check_directory(self.global_ckp_dir)

        self.mode = cfg.update_mode

        self.N = cfg.n_clients
        self.M = cfg.n_update_clients
        self.dataset = Dataset(cfg.data_dir, self.N)

        self.input_size = cfg.input_size
        self.output_channel = cfg.output_channel
        self.model = models.models_dict[cfg.model](
            self.input_size, self.output_channel
        ).to(self.device)
        self.global_model = copy.deepcopy(self.model)
        self.batch_size = cfg.batch_size
        self.n_rounds = cfg.n_rounds
        self.n_epochs = cfg.n_epochs
        self.lr = cfg.lr

    def test(self):
        self.global_model.eval()
        test_loss = 0
        correct = 0
        test_dataloader = self.dataset.get_test_dataloader(self.batch_size)
        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(self.device), target.squeeze().long().to(self.device)
                outputs = self.global_model(data)

                criterion = nn.CrossEntropyLoss()
                test_loss += criterion(outputs, target)

                predict = outputs.argmax(dim=1, keepdim=True)
                correct += predict.eq(target.view_as(predict)).sum().item()
        accuracy = correct / len(test_dataloader.dataset)
        return test_loss, accuracy

    def train(self):
        client_dataloaders = self.dataset.get_train_dataloaders(self.batch_size)
        best_acc = 0
        global_ckp_path = os.path.join(self.global_ckp_dir, "global.pth")
        torch.save(self.global_model.state_dict(), global_ckp_path)

        for r in range(self.n_rounds):
            if self.mode == 'all':
                idx = range(self.N)
            elif self.mode == 'partial':
                idx = np.random.choice(self.N, self.M, replace=False)
            else:
                raise Exception("Unexcepted update mode.")

            for i in tqdm(idx):
                # send global parameters in checkpoint to client
                client_model = copy.deepcopy(self.model)
                client_model.load_state_dict(torch.load(global_ckp_path))

                # client_i training at data_i
                dataloader = client_dataloaders[i]
                optimizer = optim.SGD(client_model.parameters(), lr=self.lr)

                for e in range(self.n_epochs):
                    for features, labels in dataloader:
                        features, labels = features.to(self.device), labels.squeeze().long().to(self.device)
                        optimizer.zero_grad()
                        outputs = client_model(features)
                        # loss = nn.CrossEntropyLoss(outputs, labels) 
                        criterion = nn.CrossEntropyLoss()
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                # save client model parameters
                save_path = os.path.join(self.client_ckp_dir, f"{i+1}.pth")
                torch.save(client_model.state_dict(), save_path)

            # calc avg of client parameters
            avg_model = copy.deepcopy(self.model)

            for i in idx:
                client_model_path = os.path.join(self.client_ckp_dir, f"{i+1}.pth")
                client_model = copy.deepcopy(self.model)
                client_model.load_state_dict(torch.load(client_model_path))
                for avg_param, client_param in zip(avg_model.parameters(), client_model.parameters()):
                    if i == idx[0]:
                        avg_param.data = client_param.data
                    else:
                        avg_param.data += client_param.data

            for avg_param in avg_model.parameters():
                avg_param.data /= self.M

            # test model of this round, save if it's better
            self.global_model.load_state_dict(avg_model.state_dict())
            test_loss, accuracy = self.test()

            print(f"Round {r+1}, Test Loss: {test_loss}, Accuracy: {accuracy}")

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(self.global_model.state_dict(), global_ckp_path)