import torch
import numpy as np

from model.res2unet3d import Res2UNet3D
from config import DEVICE, IN_CHANNELS, NUM_CLASSES

class ForwardPipeline:
    def __init__(self, model_path=None):
        self.model = Res2UNet3D(IN_CHANNELS, NUM_CLASSES).to(DEVICE)
        self.model.eval()

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    def predict(self, volume):
        x = torch.tensor(volume).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = self.model(x)
            pred = torch.sigmoid(pred)

        mask = pred.squeeze().cpu().numpy()
        return mask
