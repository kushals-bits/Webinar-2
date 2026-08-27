"""
Image Resizing & Dimension Standardization Module
Implements direct resizing and aspect-ratio preserved letterboxing.
"""

from PIL import Image


def resize_image_direct(img: Image.Image, target_size: tuple = (64, 64), resample=Image.Resampling.BILINEAR) -> Image.Image:
    """Directly scales image to target_size (width, height), potentially changing aspect ratio."""
    return img.resize(target_size, resample=resample)


def resize_image_aspect_preserve(
    img: Image.Image,
    target_size: tuple = (64, 64),
    bg_color: tuple = (255, 255, 255),
    resample=Image.Resampling.BILINEAR
) -> Image.Image:
    """
    Resizes image maintaining original aspect ratio by scaling down/up to fit within target_size,
    then pasting onto a centered background canvas of size target_size (letterboxing/padding).
    """
    target_w, target_h = target_size
    orig_w, orig_h = img.size

    # Calculate scale factor
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    resized_img = img.resize((new_w, new_h), resample=resample)
    
    # Create canvas and paste in center
    canvas = Image.new("RGB", target_size, color=bg_color)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized_img, (paste_x, paste_y))

    return canvas


def batch_resize_images(images: list, target_size: tuple = (64, 64), preserve_aspect: bool = True) -> list:
    """Resizes a collection of PIL Images to uniform target_size."""
    if preserve_aspect:
        return [resize_image_aspect_preserve(img, target_size=target_size) for img in images]
    else:
        return [resize_image_direct(img, target_size=target_size) for img in images]
