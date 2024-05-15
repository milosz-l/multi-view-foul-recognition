import lightning as L
import torch
import os


from SoccerNet.Evaluation.MV_FoulRecognition import evaluate
from torch.utils.data import DataLoader, random_split

from src.training import TrainingConfig
from src.dataset import MultiViewDataset
from src.model import LitMVNNetwork, get_pre_model
from src.loss import get_criterion
from src.eval import save_evaluation_file
from datetime import datetime
import os
from src.augment import get_augmentation
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

wandb.login()

torch.cuda.empty_cache()

num_epochs = 40
start_frame = 0
end_frame = 115
fps = 25
num_views = 5
pre_model = "s3d"
max_num_worker_train = 4
max_num_worker_val = 4
max_num_worker_test = 0
max_num_worker_chall = 0
batch_size = 4
data_aug = False
pooling_type = 'max'
weight_decay = 0.001
step_size = 3
gamma = 0.1
LR = 0.01
weighted_loss = True
data_aug = True

training_config = TrainingConfig(start_frame=start_frame, end_frame=end_frame, fps=fps, num_views = num_views, pre_model = pre_model,
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

train_size = int(0.7 * len(dataset_Train))
val_size = len(dataset_Train) - train_size

train_set, val_set = random_split(dataset_Train, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=max_num_worker_train, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=max_num_worker_val, pin_memory=True) 

dataset_Test = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views = 5, 
transform_model=transforms_model)


dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Chall', num_views = 5, 
        transform_model=transforms_model)

test_loader = torch.utils.data.DataLoader(dataset_Test,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker_test, pin_memory=False)
        
chall_loader = torch.utils.data.DataLoader(dataset_Chall,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker_chall, pin_memory=False)

criterion = get_criterion(weighted_loss, dataset_train=dataset_Train)
model = LitMVNNetwork(pre_model=pre_model, pooling_type=pooling_type, criterion=criterion, config=training_config)
job_id = str(datetime.now())
wand_logger = WandbLogger(log_model="all")

os.makedirs(f"/net/tscratch/people/{username}/lightning_logs", exist_ok=True)


checkpoint_callback = ModelCheckpoint(dirpath=f"/net/tscratch/people/{username}/lightning_log")

trainer = L.Trainer(max_epochs=num_epochs, logger=wand_logger, strategy="ddp", num_nodes=1, default_root_dir=f"/net/tscratch/people/{username}/lightning_log")
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

os.makedirs(predictions_output_dir, exist_ok=True)

test_set = f"test_{job_id}"
chall_set = f"chall_{job_id}"

test_prediction_file = save_evaluation_file(test_loader, model=model, set_name=test_set, output_dir=predictions_output_dir)
chall_prediction_file = save_evaluation_file(chall_loader, model=model, set_name=chall_set, output_dir=predictions_output_dir)

test_results = evaluate(os.path.join(path, "Test", "annotations.json"), test_prediction_file)
wandb.log(test_results)

wandb.finish()