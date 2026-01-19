import numpy as np
import scipy.ndimage as ndimage

from config import HU_MIN, HU_MAX, PATCH_SIZE

def hu_normalize(volume):
    volume = np.clip(volume, HU_MIN, HU_MAX)
    volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
    return volume.astype(np.float32)

def resize_volume(volume, target_shape):
    factors = [
        target_shape[0] / volume.shape[0],
        target_shape[1] / volume.shape[1],
        target_shape[2] / volume.shape[2]
    ]
    volume = ndimage.zoom(volume, factors, order=1)
    return volume

def center_crop(volume, patch_size=PATCH_SIZE):
    d, h, w = volume.shape
    pd, ph, pw = patch_size

    sd = (d - pd) // 2
    sh = (h - ph) // 2
    sw = (w - pw) // 2

    return volume[sd:sd+pd, sh:sh+ph, sw:sw+pw]

def preprocess(volume):
    volume = hu_normalize(volume)
    volume = center_crop(volume)
    return volume
