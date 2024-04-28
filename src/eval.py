from src.config import INVERSE_EVENT_DICTIONARY
import torch
import pytorch_lightning as L
import os
import json
import gc
from tqdm import tqdm
from typing import Union

def save_evaluation_file(dataloader,
          model: Union[torch.nn.Module, L.LightningModule],
          set_name="test",
          output_dir="."
        ):
    

    model.eval()

    prediction_file = os.path.join(output_dir, "predicitions_" + set_name + ".json")
    data = {}
    data["Set"] = set_name

    actions = {}
           
    for _, _, mvclips, action in tqdm(dataloader):

        mvclips = mvclips.cuda().float()
        outputs_offence_severity, outputs_action, _ = model(mvclips)

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


        gc.collect()
        torch.cuda.empty_cache()
    
    data["Actions"] = actions

    with open(prediction_file, "w") as outfile: 
        json.dump(data, outfile)  
    return prediction_file
