
from pathlib import Path
from PIL import Image, ImageOps


def load_and_prepare_image(image_path):
    """
    Load an image, correct its EXIF orientation,
    and convert it to RGB.

    Returns:
        PIL.Image.Image: Prepared RGB image.
    """
    image_path = Path(image_path)

    with Image.open(image_path) as img:
        # Correct orientation using EXIF metadata
        img = ImageOps.exif_transpose(img)

        # Ensure consistent RGB input
        img = img.convert("RGB")

        # Return an independent image after closing the file
        return img.copy()


def preprocess_image_for_clip(image_path, processor):
    """
    Prepare an image for the CLIP model.

    Returns:
        torch.Tensor: CLIP pixel values.
    """
    image = load_and_prepare_image(image_path)

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    return inputs["pixel_values"]
