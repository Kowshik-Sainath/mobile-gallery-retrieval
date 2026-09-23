"""
Data Utilities and Image Transforms.
Adapted from torchvision & open_clip standard pre-processing pipelines.
"""

from torchvision import transforms
from PIL import Image

def get_transforms(image_size=224, is_train=False):
    """
    Returns standard image and sketch preprocessing transforms for MobileCLIP.
    MobileCLIP models are trained on [0, 1] tensor inputs (image_mean=(0,0,0), image_std=(1,1,1)).
    Standard ImageNet normalization is NOT used in MobileCLIP and causes representation collapse.
    """
    if is_train:
        photo_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
        ])

        sketch_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.96, 1.04), fill=255),
            transforms.RandomPerspective(distortion_scale=0.08, p=0.3, fill=255),
            transforms.ToTensor(),
        ])
    else:
        photo_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        sketch_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    target_sketch_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    return photo_transform, sketch_transform, target_sketch_transform

