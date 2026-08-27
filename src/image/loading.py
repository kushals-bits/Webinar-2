"""
Image Loading & Dataset Management Module
Loads images from directories, ensures consistent color channels (RGB), and handles batch structures.
"""

import os
import numpy as np
from PIL import Image


def load_image(file_path: str, target_mode: str = "RGB") -> Image.Image:
    """Loads an image file and converts to target color mode (default RGB)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image not found at path: {file_path}")
    with Image.open(file_path) as img:
        return img.convert(target_mode)


def load_image_dataset(root_dir: str = "data/image/raw"):
    """
    Scans a class-based directory structure (root_dir/class_name/image_name.png),
    loads images, and returns:
    - raw_images: list of PIL.Image objects
    - labels: list of class names
    - file_paths: list of file paths
    - class_to_idx: dictionary mapping class name to integer ID
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Root image directory not found at: {root_dir}")

    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    raw_images = []
    labels = []
    file_paths = []

    for cls in classes:
        cls_dir = os.path.join(root_dir, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                fpath = os.path.join(cls_dir, fname)
                img = load_image(fpath, target_mode="RGB")
                raw_images.append(img)
                labels.append(cls)
                file_paths.append(fpath)

    print(f"Loaded {len(raw_images)} images across {len(classes)} classes: {classes}")
    return raw_images, np.array(labels), file_paths, class_to_idx
