import lightning as L
import torch
from typing import Literal

from src.loss import calculate_loss, calculate_outputs
from src.mvnetwork import MVNetwork
from src.training import TrainingConfig

import torchvision.transforms as transforms
from torchvision.models.video import (MC3_18_Weights, MViT_V1_B_Weights,
                                      MViT_V2_S_Weights, R2Plus1D_18_Weights,
                                      R3D_18_Weights, S3D_Weights, mvit_v1_b,
                                      mvit_v2_s)

class LitMVNNetwork(L.LightningModule):
    def __init__(self, pre_model, pooling_type, criterion, config: TrainingConfig):
        super().__init__()
        self.model = MVNetwork(net_name=pre_model, agr_type=pooling_type)
        # TODO - replace with config
        self.LR = config.LR
        self.weight_decay = config.weight_decay
        self.step_size = config.step_size
        self.gamma = config.gamma

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
        outputs_offence_severity, outputs_action, actions = calculate_outputs(
            outputs_offence_severity, outputs_action, action, self.actions)
        self.actions = actions
        
        if len(outputs_offence_severity.shape) == 1 and len(targets_offence_severity.shape) == 2:
            outputs_offence_severity = outputs_offence_severity.reshape(targets_offence_severity.shape[0], -1)
        if len(outputs_action.shape) == 1 and len(targets_action.shape) == 2:
            outputs_action = outputs_action.reshape(targets_action.shape[0], -1)

        loss = calculate_loss(self.criterion, outputs_offence_severity,
                              outputs_action, targets_offence_severity, targets_action)
        self.log("train_step_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("train_epoch_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss
    
    def forward(self, mvclips: torch.Tensor) -> torch.Any:
        mvclips = mvclips.to(self.device)
        outputs_offence_severity, outputs_action, _ = self.model(mvclips)
        return outputs_offence_severity, outputs_action, _
    
    def validation_step(self, batch, batch_idx):
        targets_offence_severity, targets_action, mvclips, action = batch
        outputs_offence_severity, outputs_action, _ = self.model(mvclips)
        outputs_offence_severity, outputs_action, actions = calculate_outputs(
            outputs_offence_severity, outputs_action, action, self.actions)
        self.actions = actions
        
        if len(outputs_offence_severity.shape) == 1 and len(targets_offence_severity.shape) == 2:
            outputs_offence_severity = outputs_offence_severity.reshape(targets_offence_severity.shape[0], -1)
        if len(outputs_action.shape) == 1 and len(targets_action.shape) == 2:
            outputs_action = outputs_action.reshape(targets_action.shape[0], -1)

        loss = calculate_loss(self.criterion, outputs_offence_severity,
                              outputs_action, targets_offence_severity, targets_action)
        self.log("val_step_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("val_epoch_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

def get_pre_model(pre_model: Literal["r3d_18", "s3d", "mc3_18", "r2plus1d_18", "mvit_v2_s"]):
    if pre_model == "r3d_18":
        transforms_model = R3D_18_Weights.KINETICS400_V1.transforms()
    elif pre_model == "s3d":
        transforms_model = S3D_Weights.KINETICS400_V1.transforms()
    elif pre_model == "mc3_18":
        transforms_model = MC3_18_Weights.KINETICS400_V1.transforms()
    elif pre_model == "r2plus1d_18":
        transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
    elif pre_model == "mvit_v2_s":
        transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    
    return transforms_model