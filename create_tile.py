import os
import random
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms import ColorJitter, GaussianBlur
import argparse
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import utils


def random_crop(image, crop_size: tuple[int, int]):
    cropper = T.RandomCrop(size=crop_size)
    return cropper(image)


def apply_random_filter(image, output_size: tuple[int, int]):
    filters = [  # ここは自由に
        T.Compose(
            [
                T.Resize(64),
                T.Resize(size=output_size),
            ]
        ),
        T.Compose(
            [
                T.Resize(32),
                T.Resize(output_size),
            ]
        ),
    ]

    filter = T.RandomChoice(filters)
    return filter(image)


def process_images(args):
    input_dir = Path(args.input_dir)
    caption_input_dir = args.caption_input_dir
    if caption_input_dir is None:
        caption_input_dir = input_dir
    else:
        caption_input_dir = Path(caption_input_dir)
    cropped_output_dir = Path(args.cropped_output_dir)
    filtered_output_dir = Path(args.filtered_output_dir)
    caption_output_dir = Path(args.caption_output_dir)
    crop_num = args.crop_num
    crop_size = tuple(args.crop_size)
    min_size = args.min_size
    max_images = args.max_images
    threads = args.threads

    # mkdir
    cropped_output_dir.mkdir(parents=True, exist_ok=True)
    filtered_output_dir.mkdir(parents=True, exist_ok=True)
    caption_output_dir.mkdir(parents=True, exist_ok=True)

    images = utils.glob_all_images(input_dir)
    # 最近のものを始めにおく(数字が大きいものが先)
    images = list(reversed(sorted(images, key=lambda x: x.stem.zfill(10))))

    # print(images[:10])

    def process_chunk(images, max_count, pbar):
        total_images = 0
        for img_path in images:
            if max_images is not None and total_images >= max_count:
                break

            image = Image.open(img_path).convert("RGB")

            if max(image.size) < min_size:
                continue

            for i in range(crop_num):
                cropped_img = random_crop(image, crop_size)
                filtered_img = apply_random_filter(cropped_img, output_size=crop_size)

                stem = img_path.stem
                suffix = img_path.suffix

                cropped_output_path = cropped_output_dir / f"{stem}_{i}{suffix}"
                filtered_output_path = filtered_output_dir / f"{stem}_{i}{suffix}"

                cropped_img.save(cropped_output_path)
                filtered_img.save(filtered_output_path)

                caption_path = caption_input_dir / f"{stem}.txt"
                shutil.copy(caption_path, caption_output_dir / f"{stem}_{i}.txt")

                total_images += 1
                pbar.update(1)

    chunks = np.array_split(images, threads)
    max_count = max_images // threads

    with tqdm(total=max_images) as pbar:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(process_chunk, chunk, max_count, pbar))
            for future in futures:
                future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images by cropping and applying filters."
    )
    parser.add_argument(
        "--input_dir", "-i", required=True, help="Input directory containing images."
    )
    parser.add_argument(
        "--caption_input_dir",
        help="Input directory containing captions.",
    )

    parser.add_argument(
        "--cropped_output_dir",
        required=True,
        help="Output directory for cropepd images.",
    )
    parser.add_argument(
        "--filtered_output_dir",
        required=True,
        help="Output directory for processed images.",
    )
    parser.add_argument(
        "--caption_output_dir",
        required=True,
        help="Output directory for captions.",
    )

    parser.add_argument(
        "--crop_size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Size for cropped images.",
    )
    parser.add_argument(
        "--crop_num", type=int, default=5, help="Number of crops to make per image."
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=300,
        help="Minimum size for images to be processed.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to be processed.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=100,
        help="Threads to use for processing.",
    )

    args = parser.parse_args()

    process_images(args)
