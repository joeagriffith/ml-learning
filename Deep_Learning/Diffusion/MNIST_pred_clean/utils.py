import matplotlib.pyplot as plt
import torch
from tqdm.auto import trange

# Takes a single image and shows it
def showImage(img):
    plt.imshow(img.squeeze().cpu(), cmap='gray')
    plt.axis('off')
    plt.show()

# Takes a list of images and plots them in a row
def showImages(imgs):
    fig, axes = plt.subplots(1, len(imgs), figsize=(15,5))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].squeeze().cpu(), cmap='gray')
        ax.axis('off')
    plt.show()

# Takes 16 images and plots them in a 4x4 grid
def showExamples(imgs):
    assert len(imgs) == 16, "Number of images must be 16"
    plt.figure(figsize=(8,8))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(imgs[i].squeeze().cpu(), cmap='gray')
        plt.axis('off')
    plt.show()