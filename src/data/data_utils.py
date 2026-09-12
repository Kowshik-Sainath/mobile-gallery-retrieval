"""
Data Utilities and Image Transforms.
Adapted from torchvision & open_clip standard pre-processing pipelines.
"""

from torchvision import transforms
from PIL import Image

def get_transforms(image_size=224):
    """
    Returns standard image and sketch preprocessing transforms.
    """
    # Standard CLIP / MobileCLIP Image Normalization
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    photo_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    sketch_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    target_sketch_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    return photo_transform, sketch_transform, target_sketch_transform
