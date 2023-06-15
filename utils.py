from pathlib import Path


def glob_all_images(path: str | Path):
    path = Path(path)
    images = list(path.glob("*"))
    images = [
        image
        for image in images
        if image.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]
    ]
    return images
