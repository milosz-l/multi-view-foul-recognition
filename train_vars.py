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
from src.training import TrainingConfig

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

# num_epochs = 2
# start_frame = 41
# end_frame = 99
# fps = 15
# num_views = 5
# pre_model = "swin3d"
# max_num_worker_train = 2
# max_num_worker_val = 2
num_epochs = 20
start_frame = 58
end_frame = 92
fps = 12
num_views = 5
pre_model = "mvit_v2_s"
max_num_worker_train = 4
max_num_worker_val = 4
max_num_worker_test = 4
max_num_worker_chall = 4
batch_size = 2
pooling_type = 'attention'
weight_decay = 0.001
step_size = 3
gamma = 0.3
LR = 5e-5
weighted_loss = "Yes"   # "Yes" or anything else
data_aug = True         # bool value
load_pretrained_model = False

training_config = TrainingConfig(start_frame=start_frame, end_frame=end_frame, fps=fps, num_views=num_views, pre_model=pre_model,
                                 max_num_worker=max_num_worker_train, batch_size=batch_size, data_aug=data_aug, pooling_type=pooling_type,
                                 weight_decay=weight_decay, step_size=step_size, gamma=gamma, LR=LR, weighted_loss=weighted_loss)

# Get the current username
username = os.environ['USER']

# Use the username to construct paths
path = f"/net/tscratch/people/{username}/data"
predictions_output_dir = f"/net/tscratch/people/{username}/outputs"

transform_aug = get_augmentation(training_config.data_aug)
transforms_model = get_pre_model(training_config.pre_model)


dataset_Train = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Train',
                                 num_views=5, transform=transform_aug, transform_model=transforms_model)

dataset_Valid2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Valid', num_views = 5, 
    transform_model=transforms_model)

train_loader = torch.utils.data.DataLoader(dataset_Train,
    batch_size=batch_size, shuffle=True,
    num_workers=max_num_worker_train, pin_memory=True)

val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
    batch_size=1, shuffle=False,
    num_workers=max_num_worker_val, pin_memory=True)

dataset_Test = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views=5,
                                transform_model=transforms_model)


dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Chall', num_views=5,
                                 transform_model=transforms_model)

test_loader = DataLoader(dataset_Test,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker_test, pin_memory=False)
        
chall_loader = DataLoader(dataset_Chall,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker_chall, pin_memory=False)

criterion = get_criterion(weighted_loss, dataset_train=dataset_Train)
model = LitMVNNetwork(pre_model=pre_model, pooling_type=pooling_type, criterion=criterion, config=training_config, test_loader=test_loader, chall_loader=chall_loader).cuda()

if load_pretrained_model:
    # Load pretrained weights if available
    path_to_model_weights = "14_model.pth.tar"
    if os.path.exists(path_to_model_weights):
        load = torch.load(path_to_model_weights)
        model.load_state_dict(load['state_dict'])

job_id = str(datetime.now())
wandb.finish()
wand_logger = WandbLogger(project="ZZSN multi-view-foul-recognition", config=training_config.model_dump(), log_model="all")

os.makedirs(f"/net/tscratch/people/{username}/lightning_logs", exist_ok=True)


checkpoint_callback = ModelCheckpoint(
    dirpath=f"/net/tscratch/people/{username}/lightning_log")

trainer = L.Trainer(max_epochs=num_epochs, logger=wand_logger, num_nodes=1, default_root_dir=f"/net/tscratch/people/{username}/lightning_log")
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader2)

os.makedirs(predictions_output_dir, exist_ok=True)

test_set = f"test_{job_id}"
chall_set = f"chall_{job_id}"

torch.cuda.empty_cache()
test_prediction_file = save_evaluation_file(test_loader, model=model, set_name=test_set, output_dir=predictions_output_dir)
torch.cuda.empty_cache()
chall_prediction_file = save_evaluation_file(chall_loader, model=model, set_name=chall_set, output_dir=predictions_output_dir)

test_results = evaluate(os.path.join(
    path, "Test", "annotations.json"), test_prediction_file)
wandb.log(test_results)

wandb.finish()
