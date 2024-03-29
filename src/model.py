import lightning as L
import torch
from torch import Tensor, nn, optim, utils
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.loss import calculate_loss, calculate_outputs
from src.mvnetwork import MVNetwork


class LitMVNNetwork(L.LightningModule):
    def __init__(self, pre_model, pooling_type, criterion):
        super().__init__()
        self.model = MVNetwork(net_name=pre_model, agr_type=pooling_type)
        # TODO - replace with config
        self.LR = 0.01
        self.weight_decay = 0.001
        self.step_size = 3
        self.gamma = 0.1

        self.criterion = criterion
        self.actions = {}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.LR,
                                      betas=(0.9, 0.999), eps=1e-07,
                                      weight_decay=self.weight_decay, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], {"scheduler": scheduler, "interval": "epoch"}

    def training_step(self, batch, batch_idx):
        targets_offence_severity, targets_action, mvclips, action = batch
        outputs_offence_severity, outputs_action, _ = self.model(mvclips)
        print("x")
        outputs_offence_severity, outputs_action, actions = calculate_outputs(
            outputs_offence_severity, outputs_action, action, self.actions)
        print("x")
        self.actions = actions
        loss = calculate_loss(self.criterion, outputs_offence_severity,
                              outputs_action, targets_offence_severity, targets_action)
        print("x")
        return loss
