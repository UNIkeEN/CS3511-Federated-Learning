import sys
import socket
import io
import os
import dill
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import logging
from datetime import datetime

import models
from pipeline import atom_train
from utils import receive_data

def client_process(client_id, cfg):
    with open(os.path.join(cfg.data_dir, f"Client{client_id}.pkl"), "rb") as f:
        private_data = dill.load(f)
        private_dataloader = DataLoader(
            private_data,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True
        )
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((cfg.server_address, int(cfg.port)))
    client_model = models.models_dict[cfg.model](cfg.input_size, cfg.output_channel).to(cfg.device)

    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    log_path = os.path.join(cfg.log_dir, f"client_{client_id}_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
    logging.basicConfig(
        level=logging.DEBUG,
        format = '%(asctime)s [%(levelname)s] %(message)s',
        handlers = [
            logging.FileHandler(log_path), 
            logging.StreamHandler()
        ])

    while True:
        try:
            # recv global state
            data = receive_data(client_socket, cfg.buffer_size)
            if data == b"FIN":
                logging.info(f"Received end signal. Closing client {client_id}.")
                break
            elif data is None:
                logging.error(f"No data received, connection may be closed. Closing client {client_id}.")
                break
            
            buffer = io.BytesIO(data)
            buffer.seek(0)
            global_model_state = torch.load(buffer)
            
            client_model.load_state_dict(global_model_state)
            atom_train(client_model, cfg.lr, private_dataloader, cfg.n_epochs, cfg.device)

            # send client state
            buffer = io.BytesIO()
            torch.save(client_model.state_dict(), buffer)
            buffer.seek(0)
            client_socket.sendall(buffer.getvalue())
            client_socket.sendall(b"END")
        except socket.error as e:
            logging.error(f"Socket error for client {client_id}: ", e)
            break

    client_socket.close()

if __name__ == "__main__":
    client_id, cfg_file = sys.argv[1:3]
    cfg = OmegaConf.load(cfg_file)

    client_process(client_id, cfg)
