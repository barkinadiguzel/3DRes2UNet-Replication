import matplotlib.pyplot as plt
import numpy as np

def show_slice(volume, mask, slice_idx=None):
    if slice_idx is None:
        slice_idx = volume.shape[0] // 2

    img = volume[slice_idx]
    msk = mask[slice_idx]

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title("CT Slice")
    plt.imshow(img, cmap="gray")

    plt.subplot(1,2,2)
    plt.title("Segmentation Overlay")
    plt.imshow(img, cmap="gray")
    plt.imshow(msk, alpha=0.5, cmap="jet")

    plt.show()
