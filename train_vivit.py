import os
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from torch.utils.data import DataLoader, random_split

import wandb
from src.augment import get_augmentation
from src.dataset import MultiViewDataset
from src.eval import save_evaluation_file
from src.loss import get_criterion
from src.model import LitMVNNetwork, get_pre_model
from src.training import MVNTrainingConfig

num_epochs = 40
start_frame = 0
end_frame = 8
fps = 25
num_views = 5
pre_model = "s3d"
max_num_worker_train = 4
max_num_worker_val = 4
max_num_worker_test = 3
max_num_worker_chall = 0
batch_size = 4
data_aug = False
pooling_type = 'max'
weight_decay = 0.001
step_size = 3
gamma = 0.1
LR = 0.01
weighted_loss = False
data_aug = True

training_config = ViVTTrainingConfig(start_frame=start_frame, end_frame=end_frame, fps=fps, num_views=num_views, pre_model=pre_model,
                                     max_num_worker=max_num_worker_train, batch_size=batch_size, data_aug=data_aug, pooling_type=pooling_type,
                                     weight_decay=weight_decay, step_size=step_size, gamma=gamma, LR=LR, weighted_loss=weighted_loss)


wandb.init(
    project="ZZSN multi-view-foul-recognition",
    config=training_config.model_dump()
)


# Get the current username
username = os.environ['USER']

# Use the username to construct paths
path = f"/net/tscratch/people/{username}/data"
predictions_output_dir = f"/net/tscratch/people/{username}/outputs"

transform_aug = get_augmentation(training_config.data_aug)
transforms_model = get_pre_model(training_config.pre_model)

dataset_Train = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Train',
                                 num_views=5, transform=transform_aug, transform_model=transforms_model)
