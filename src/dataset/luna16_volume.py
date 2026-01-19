import SimpleITK as sitk
import numpy as np

from .preprocessing import preprocess

class LUNA16Volume:
    def __init__(self, mhd_path):
        self.mhd_path = mhd_path

    def load(self):
        itk_image = sitk.ReadImage(self.mhd_path)
        volume = sitk.GetArrayFromImage(itk_image)  
        volume = preprocess(volume)
        return volume
