from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from src.config import INVERSE_EVENT_DICTIONARY


def get_criterion(weighted_loss: bool, dataset_train: Dataset):
    if weighted_loss == 'Yes':
        criterion_offence_severity = nn.CrossEntropyLoss(
            weight=dataset_train.getWeights()[0].cuda())
        criterion_action = nn.CrossEntropyLoss(
            weight=dataset_train.getWeights()[1].cuda())
    else:
        criterion_offence_severity = nn.CrossEntropyLoss()
        criterion_action = nn.CrossEntropyLoss()
    criterion = (criterion_offence_severity, criterion_action)
    return criterion


def calculate_outputs(outputs_offence_severity, outputs_action, action, actions) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(action) == 1:
        preds_sev = torch.argmax(outputs_offence_severity, 0)
        preds_act = torch.argmax(outputs_action, 0)

        values = {}
        values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
        if preds_sev.item() == 0:
            values["Offence"] = "No offence"
            values["Severity"] = ""
        elif preds_sev.item() == 1:
            values["Offence"] = "Offence"
            values["Severity"] = "1.0"
        elif preds_sev.item() == 2:
            values["Offence"] = "Offence"
            values["Severity"] = "3.0"
        elif preds_sev.item() == 3:
            values["Offence"] = "Offence"
            values["Severity"] = "5.0"
        actions[action[0]] = values
    else:
        preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
        preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

        for i in range(len(action)):
            values = {}
            values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
            if preds_sev[i].item() == 0:
                values["Offence"] = "No offence"
                values["Severity"] = ""
            elif preds_sev[i].item() == 1:
                values["Offence"] = "Offence"
                values["Severity"] = "1.0"
            elif preds_sev[i].item() == 2:
                values["Offence"] = "Offence"
                values["Severity"] = "3.0"
            elif preds_sev[i].item() == 3:
                values["Offence"] = "Offence"
                values["Severity"] = "5.0"
            actions[action[i]] = values

            if len(outputs_offence_severity.size()) == 1:
                outputs_offence_severity = outputs_offence_severity.unsqueeze(
                    0)
            if len(outputs_action.size()) == 1:
                outputs_action = outputs_action.unsqueeze(0)

    return outputs_offence_severity, outputs_action, actions


def calculate_loss(criterion, outputs_offence_severity, outputs_action, targets_offence_severity, targets_action):

    loss_offence_severity = criterion[0](
        outputs_offence_severity, targets_offence_severity)
    loss_action = criterion[1](outputs_action, targets_action)
    loss = loss_offence_severity + loss_action
    return loss
