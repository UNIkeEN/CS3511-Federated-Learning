import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
import subprocess
import socket
import threading
import io

import models
from dataset import Dataset
from utils import check_directory, receive_data

class Pipeline(ABC):
    def __init__(self, cfg):
        self.device = cfg.device
        self.client_ckp_dir = cfg.client_ckp_dir
        self.global_ckp_dir = cfg.global_ckp_dir
        check_directory(self.client_ckp_dir)
        check_directory(self.global_ckp_dir)

        self.mode = cfg.update_mode

        self.N = cfg.n_clients
        self.M = cfg.n_update_clients

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

        self.test_dataloader = None # subclass need to assign it.
    
    @abstractmethod
    def send_and_train(self, idx, global_state):
        """
        send global model's state to client_{id} and start training at client.
        """
        raise NotImplementedError
    
    @abstractmethod
    def recv_and_merge(self, idx):
        """
        receive client models' state and do aggregation
        """
        raise NotImplementedError
    
    def train(self):
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
            
            # send global_model and train at client
            global_state = torch.load(global_ckp_path)
            self.send_and_train(idx, global_state)                

            # calc avg of client parameters
            avg_model = self.recv_and_merge(idx)
            
            # test model of this round, save if it's better
            self.global_model.load_state_dict(avg_model.state_dict())
            test_loss, accuracy = test(self.global_model, self.test_dataloader, self.device)
            print(f"Round {r+1}, Test Loss: {test_loss}, Accuracy: {accuracy}")

            if accuracy > best_acc:
                best_acc = accuracy
                torch.save(self.global_model.state_dict(), global_ckp_path)

class OfflinePipeline(Pipeline):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.dataset = Dataset(cfg.data_dir, self.N)
        self.client_dataloaders = self.dataset.get_train_dataloaders(self.batch_size)
        self.test_dataloader = self.dataset.get_test_dataloader(self.batch_size)

    def send_and_train(self, idx, global_state):
        client_model = copy.deepcopy(self.model)
        client_model.load_state_dict(global_state)

        # client_i training at data_i
        for i in tqdm(idx):
            dataloader = self.client_dataloaders[i]
            atom_train(client_model, self.lr, dataloader, self.n_epochs, self.device)
            
            save_path = os.path.join(self.client_ckp_dir, f"{i+1}.pth")
            torch.save(client_model.state_dict(), save_path)

    def recv_and_merge(self, idx):
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
        
        return avg_model
    
class OnlinePipeline(Pipeline):
    def __init__(self, cfg, config_path):
        super().__init__(cfg)
        self.dataset = Dataset(cfg.data_dir, self.N, load_train=False)
        self.test_dataloader = self.dataset.get_test_dataloader(self.batch_size)
        
        self.port = cfg.port
        self.server_address = cfg.server_address
        self.buffer_size = cfg.buffer_size

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.server_address, self.port))
        self.server_socket.listen(self.N)

        # start client process
        for i in range(self.N):
            subprocess.Popen(["python", "client.py", str(i+1), str(config_path)])
            # command = f'start cmd.exe /k python client.py {i+1} {config_path}'
            # subprocess.Popen(command, shell=True)
            print(f"Client {i+1} started")
        
        # connect client
        self.client_sockets = []
        self.lock = threading.Lock()
        for _ in range(self.N):
            client_socket, addr = self.server_socket.accept()
            self.client_sockets.append(client_socket)
            print(f"Connected to client at {addr}")

    def send_and_train(self, idx, global_state):
        threads = []
        # import time
        for i in idx:
            # time.sleep(1)
            thread = threading.Thread(target=self._send_model, args=(self.client_sockets[i], global_state, i))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def _send_model(self, client_socket, global_state, i):
        try:
            buffer = io.BytesIO()
            torch.save(global_state, buffer)
            buffer.seek(0)
            client_socket.sendall(buffer.getvalue())
            client_socket.sendall(b"END")
            print(f"Send model to client {i}")
        except socket.error as e:
            print("Socket error during sending model:", e)

    def recv_and_merge(self, idx):
        threads = []
        self.avg_model = copy.deepcopy(self.model)

        for i in idx:
            thread = threading.Thread(target=self._recv_model, args=(self.client_sockets[i], i, idx))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        for avg_param in self.avg_model.parameters():
            avg_param.data /= self.M
        
        return self.avg_model

    def _recv_model(self, client_socket, i, idx):
        try:
            data = receive_data(client_socket, self.buffer_size)
            if data is None:
                print("Socket error during receiving client model")
            
            buffer = io.BytesIO(data)
            buffer.seek(0)
            client_model_state = torch.load(buffer)

            client_model = copy.deepcopy(self.model)
            client_model.load_state_dict(client_model_state)
            save_path = os.path.join(self.client_ckp_dir, f"{i+1}.pth")
            torch.save(client_model.state_dict(), save_path)
            with self.lock:
                for avg_param, client_param in zip(self.avg_model.parameters(), client_model.parameters()):
                    if i == idx[0]:
                        avg_param.data = client_param.data
                    else:
                        avg_param.data += client_param.data
            print(f"Recv model from client {i}")
        except socket.error as e:
            print("Socket error during receiving client model:", e)
    
    def train(self):
        super().train()

        # send end msg to client
        for client_socket in self.client_sockets:
            try:
                client_socket.sendall(b"FIN")
                client_socket.close()
            except socket.error as e:
                print(f"Error sending end signal: {e}")
    
def atom_train(model, lr, dataloader, n_epochs, device):
    """
    train at one client
    """
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for e in range(n_epochs):
        for features, labels in dataloader:
            features, labels = features.to(device), labels.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(features)
            # loss = nn.CrossEntropyLoss(outputs, labels) 
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test(model, test_dataloader, device):
    """
    test model's accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.squeeze().long().to(device)
            outputs = model(data)

            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(outputs, target)

            predict = outputs.argmax(dim=1, keepdim=True)
            correct += predict.eq(target.view_as(predict)).sum().item()
    accuracy = correct / len(test_dataloader.dataset)
    return test_loss, accuracy