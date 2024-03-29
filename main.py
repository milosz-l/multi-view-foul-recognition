import lightning as L
import torch
import torchvision.transforms as transforms
from torchvision.models.video import (MC3_18_Weights, MViT_V1_B_Weights,
                                      MViT_V2_S_Weights, R2Plus1D_18_Weights,
                                      R3D_18_Weights, S3D_Weights, mvit_v1_b,
                                      mvit_v2_s)

from src.dataset import MultiViewDataset
from src.loss import calculate_loss, calculate_outputs, get_criterion
from src.model import LitMVNNetwork
from src.mvnetwork import MVNetwork

path = "data/mvfouls"
start_frame = 0
end_frame = 125  # TODO - make lower
fps = 25
num_views = 5
pre_model = "s3d"
max_num_worker = 0
batch_size = 2
data_aug = False
pooling_type = 'max'
weight_decay = 0.001
step_size = 3
gamma = 0.1
LR = 0.01
weighted_loss = False

if data_aug == 'Yes':
    transformAug = transforms.Compose([
                                      transforms.RandomAffine(
                                          degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
                                      transforms.RandomPerspective(
                                          distortion_scale=0.3, p=0.5),
                                      transforms.RandomRotation(degrees=5),
                                      transforms.ColorJitter(
                                          brightness=0.5, saturation=0.5, contrast=0.5),
                                      transforms.RandomHorizontalFlip()
                                      ])
else:
    transformAug = None

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

dataset_Train = MultiViewDataset(path=path, start=start_frame, end=64, fps=fps, split='train',
                                 num_views=2, transform=transformAug, transform_model=transforms_model)
criterion = get_criterion(weighted_loss, dataset_Train)


train_loader = torch.utils.data.DataLoader(dataset_Train,
                                           batch_size=batch_size, shuffle=False,
                                           num_workers=max_num_worker)

model = LitMVNNetwork(pre_model=pre_model,
                      pooling_type=pooling_type, criterion=criterion)

trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=model, train_dataloaders=train_loader)