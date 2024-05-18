from typing import Literal

from pydantic import BaseModel


class BaseTrainingConfig(BaseModel):
    start_frame: int = 0
    end_frame: int = 64
    fps: int = 25
    num_views: int = 5
    max_num_worker: int = 0
    batch_size: int = 2
    data_aug: bool = False
    weight_decay: float = 0.001
    step_size: int = 3
    gamma: float = 0.1
    LR: float = 0.01
    weighted_loss: bool = False
    pre_model: Literal["r3d_18", "s3d", "mc3_18",
                       "r2plus1d_18", "mvit_v2_s"] = "s3d"


class MVNTrainingConfig(BaseTrainingConfig):
    pooling_type: Literal['avg', 'max'] = 'max'


class ViVTTrainingConfig(BaseTrainingConfig):
    # TODO - add additional configs for ViVTTraining config
    image_size: int = 224
