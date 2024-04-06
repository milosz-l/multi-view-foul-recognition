import torch
from torchvision.models.video import (MC3_18_Weights, MViT_V1_B_Weights,
                                      MViT_V2_S_Weights, R2Plus1D_18_Weights,
                                      R3D_18_Weights, S3D_Weights, mc3_18,
                                      mvit_v1_b, mvit_v2_s, r2plus1d_18,
                                      r3d_18, s3d)

from src.aggregate import MVAggregate


class MVNetwork(torch.nn.Module):

    def __init__(self, net_name='r2plus1d_18', agr_type='max', lifting_net=torch.nn.Sequential()):
        super().__init__()

        self.net_name = net_name
        self.agr_type = agr_type
        self.lifting_net = lifting_net

        self.feat_dim = 512

        if net_name == "r3d_18":
            weights_model = R3D_18_Weights.DEFAULT
            network = r3d_18(weights=weights_model)
        elif net_name == "s3d":
            weights_model = S3D_Weights.DEFAULT
            network = s3d(weights=weights_model)
            self.feat_dim = 400
        elif net_name == "mc3_18":
            weights_model = MC3_18_Weights.DEFAULT
            network = mc3_18(weights=weights_model)
        elif net_name == "r2plus1d_18":
            weights_model = R2Plus1D_18_Weights.DEFAULT
            network = r2plus1d_18(weights=weights_model)
        elif net_name == "mvit_v2_s":
            weights_model = MViT_V2_S_Weights.DEFAULT
            network = mvit_v2_s(weights=weights_model)
            self.feat_dim = 400
        else:
            weights_model = R2Plus1D_18_Weights.DEFAULT
            network = r2plus1d_18(weights=weights_model)

        network.fc = torch.nn.Sequential()

        self.mvnetwork = MVAggregate(
            model=network,
            agr_type=self.agr_type,
            feat_dim=self.feat_dim,
            lifting_net=self.lifting_net,
        )

    def forward(self, mvimages):
        return self.mvnetwork(mvimages)
