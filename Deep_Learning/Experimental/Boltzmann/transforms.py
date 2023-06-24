from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np

class GaussianNoise(object):
    """
    Applies Gaussian noise to the input image. This transform returns continuous values so it should only be used with continuous Hopfield networks which are not yet implemented.

    Args:
        |  mean (float): The mean of the Gaussian distribution.
        |  std (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: The output tensor.
    """
    def __init__(self, mean=0.0, std=0.001):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        return add_gaussian_noise(img, self.mean, self.std)

class Scale(torch.nn.Module):
    """
    Scale the input tensor from [min, max] to [-1, 1].

    Args:
        |  min (float): The minimum value of the input tensor.
        |  max (float): The maximum value of the input tensor.
    """
    def __init__(self, min=0.0, max=1.0):
        super(Scale, self).__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        """
        Applies the scaling to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor, scaled to [-1, 1].
        """
        return (x - self.min) / (self.max - self.min) * 2.0 - 1.0

    def inverse(self, x):
        """
        Performs the inverse scaling to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor, scaled to [min, max].
        """
        return (x + 1.0) / 2.0 * (self.max - self.min) + self.min

# ===================================== Functional =====================================
def mask_center_column(image, width):
    """
    Sets the center column of width `width` to -1.0.

    Args:
        |  image (torch.Tensor): The input tensor.
        |  width (float): The width of the center column.

    Returns:
        torch.Tensor: The output tensor.
    """
    image = image.clone()
    image[:, :, :, image.shape[3] // 2 - int(image.shape[3] * width) // 2 : image.shape[3] // 2 + int(image.shape[3] * width) // 2] = -1.0
    return image


def mask_center_row(image, width):
    """
    Sets the center row of width `width` to -1.0.

    Args:
        |  image (torch.Tensor): The input tensor.
        |  width (float): The width of the center row.

    Returns:
        torch.Tensor: The output tensor.
    """
    image = image.clone()
    image[:, :, image.shape[2] // 2 - int(image.shape[2] * width) // 2 : image.shape[2] // 2 + int(image.shape[2] * width) // 2, :] = -1.0
    return image

def mask_lower_half(image):
    """
    Sets the lower half of the image to -1.0.

    Args:
        |  image (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The output tensor.
    """
    image = image.clone()
    image[:, :, image.shape[2] // 2 :, :] = -1.0
    return image

def add_gaussian_noise(image, mean=0.0, std=0.001):
    """
    |  Adds Gaussian noise to the input image.
    |  Should only be used with continuous Hopfield networks which are not yet implemented.

    Args:
        |  image (torch.Tensor): The input tensor.
        |  mean (float): The mean of the Gaussian distribution.
        |  std (float): The standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: The output tensor.
    """
    noise = (torch.randn(image.shape) * std + mean)
    if image.is_cuda:
        noise = noise.to(torch.device("cuda"))
    return torch.clip(image + noise, min=-1.0, max=1.0)

def add_salt_and_pepper_noise(image, p=0.05):
    """
    |  Adds salt and pepper noise to the input image. Essentially flips random pixels to -1.0 or 1.0.
    |  Only works with discrete Hopfield networks. 

    Args:
        |  image (torch.Tensor): The input tensor.
        |  p (float): The probability of flipping a pixel.

    Returns:
        torch.Tensor: The output tensor.
    """
    noise = torch.bernoulli(torch.full(image.shape, 1.0-p)) * 2.0 - 1.0
    if image.is_cuda:
        noise = noise.to(torch.device("cuda"))
    return torch.clip(image * noise, min=-1.0, max=1.0)

def downsample_and_upsample(image, scale=2):
    """
    |  Scales the input image down by a factor of `scale` and then back up to the original size.
    |  Fidelity is lost in the process.

    Args:
        |  image (torch.Tensor): The input tensor.
        |  scale (int): The scale factor.

    Returns:
        torch.Tensor: The output tensor.
    """
    image = F.interpolate(image, scale_factor=1.0 / scale, mode="nearest")    
    image = F.interpolate(image, scale_factor=scale, mode="nearest")
    return image