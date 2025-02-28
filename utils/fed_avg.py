from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fed_utils.sampler import FederatedSampler
from hydra.utils import instantiate
import quest.utils.utils as utils
from tqdm import tqdm

class FedAvg:
    """Implementation of FedAvg
    http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf
    """
    def __init__(self, cfg, model, dataset):
        self.fl_cfg = cfg.federated
        self.train_cfg = cfg.training
        self.device = cfg.device
        self.root_model = model
        self.dataset = dataset
        self.sampler = self.get_sampler(self.fl_cfg)
        self.train_loader = instantiate(
            cfg.train_dataloader, 
            dataset=self.dataset,
            sampler=self.sampler)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.train_cfg.use_amp)

    # Set up a sampler for federated learning clients
    def get_sampler(self, cfg):
        return FederatedSampler(self.dataset,
                                cfg.non_iid,
                                cfg.n_clients,
                                cfg.n_shards
                                )

    # Compute the average of model weights from multiple clients
    def average_weights(self, weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        weights_avg = copy.deepcopy(weights[0])
        for key in weights_avg.keys():
            for i in range(1, len(weights)):
                weights_avg[key] += weights[i][key]
            weights_avg[key] = torch.div(weights_avg[key], len(weights))
        return weights_avg

    # Train a single client model
    def _train_client(
        self, root_model: nn.Module, train_loader: DataLoader, client_idx: int,
    ) -> Tuple[nn.Module, float]:
        """Train a client model."""
        model = copy.deepcopy(root_model)
        model.train()
        optimizers = model.get_optimizers()
        schedulers = model.get_schedulers(optimizers)
        total_loss = []
        
        for epoch in range(self.fl_cfg.n_client_epochs):
            epoch_loss = 0.0
            
            for idx, data in enumerate(tqdm(train_loader, disable=not self.train_cfg.use_tqdm)):
                data = utils.map_tensor_to_device(data, self.device)
                
                for optimizer in optimizers:
                    optimizer.zero_grad()

                with torch.autograd.set_detect_anomaly(False):
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.train_cfg.use_amp):
                        loss, info = model.compute_loss(data)
                
                    self.scaler.scale(loss).backward()

                for optimizer in optimizers:
                    self.scaler.unscale_(optimizer)
                if self.train_cfg.grad_clip is not None:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), self.train_cfg.grad_clip
                    )
                for optimizer in optimizers:
                    self.scaler.step(optimizer)
                self.scaler.update()
                if self.train_cfg.grad_clip is not None:
                    info.update({
                        "grad_norm": grad_norm.item(),
                    })
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
            total_loss.append(epoch_loss)
            print(
                f"Client #{client_idx} | Epoch: {epoch}/{self.fl_cfg.n_client_epochs} | Loss: {epoch_loss}",
                end="\r",
            )        
        return model, sum(total_loss)/len(total_loss)

    # Perform federated learning training
    def train(self) -> None:
        """Train a server model."""
        train_losses = []
        for epoch in range(self.fl_cfg.n_epochs):
            clients_models = []
            clients_losses = []
            # Randomly select clients for training
            m = max(int(self.fl_cfg.frac * self.fl_cfg.n_clients), 1)
            idx_clients = np.random.choice(range(self.fl_cfg.n_clients), m, replace=False)
            self.root_model.train()
            for client_idx in idx_clients:
                self.train_loader.sampler.set_client(client_idx)
                client_model, client_loss = self._train_client(
                    root_model=self.root_model,
                    train_loader=self.train_loader,
                    client_idx=client_idx,
                )
                clients_models.append(client_model.state_dict())
                clients_losses.append(client_loss)
            
            # Aggregate client models and update the server model
            updated_weights = self.average_weights(clients_models)
            self.root_model.load_state_dict(updated_weights)
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)
            if (epoch + 1) % self.fl_cfg.log_every == 0:
                avg_train_loss = sum(train_losses) / len(train_losses)
                logs = {
                    "train/loss": avg_train_loss,
                    "round": epoch,
                }
                # TODO) Log on wandb ##################################################################
                # self.logger.log(logs) 
                # Print training results for the current round
                print(f"\n\nResults after {epoch + 1} rounds of training:")
                print(f"---> Avg Training Loss: {avg_train_loss}")
