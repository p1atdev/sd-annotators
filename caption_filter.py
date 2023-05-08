from typing import Optional, List, Tuple
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import argparse

DEFAULT_FILTER_TAGS = ["monochrome", "greyscale"]

BATCH_SIZE = 200


def check_contains_tags(caption_file: Path, filter_tags):
    with open(caption_file, "r") as f:
        caption = f.read()
    for filter_tag in filter_tags:
        if filter_tag in caption:
            return True
    return False


def move_caption_and_image(
    caption_file: Path,
    image_file: Path,
    caption_output_dir: Path,
    image_output_dir: Path,
):
    os.rename(caption_file, caption_output_dir / caption_file.name)
    os.rename(image_file, image_output_dir / image_file.name)


def process_captions(
    caption_files: List[Path],
    filter_tags,
    images_dir,
    caption_output_dir,
    image_output_dir,
    pbar,
):
    for caption_file in tqdm(caption_files):
        if check_contains_tags(caption_file, filter_tags):
            # search image file. image file has same stem but unknown extension
            image_file = next(images_dir.glob(f"{caption_file.stem}.*"))
            move_caption_and_image(
                caption_file, image_file, caption_output_dir, image_output_dir
            )
            # print(caption_file, image_file)
        pbar.update(1)


def main(args):
    captions_dir = args.captions_dir
    images_dir = args.images_dir
    if images_dir is None:
        images_dir = captions_dir

    caption_output_dir = Path(args.caption_output_dir)
    image_output_dir = Path(args.image_output_dir)

    filter_tags = args.filter_tags

    caption_files = list(captions_dir.glob("*.txt"))

    print("Total captions:", len(caption_files))

    if not caption_output_dir.exists():
        caption_output_dir.mkdir()

    if not image_output_dir.exists():
        image_output_dir.mkdir()

    chunks = np.array_split(caption_files, BATCH_SIZE)

    with tqdm(total=len(caption_files)) as pbar:
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = []
            for chunk in chunks:
                futures.append(
                    executor.submit(
                        process_captions,
                        chunk,
                        images_dir,
                        filter_tags,
                        caption_output_dir,
                        image_output_dir,
                        pbar,
                    )
                )

            for future in futures:
                future.result()

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captions_dir",
        type=str,
        required=True,
        help="Path to directory containing captions",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--caption_output_dir",
        type=str,
        help="Path to directory to save captions",
    )
    parser.add_argument(
        "--image_output_dir",
        type=str,
        help="Path to directory to save images",
    )
    parser.add_argument(
        "--filter_tags",
        type=str,
        nargs="+",
        default=DEFAULT_FILTER_TAGS,
        help="Tags to filter",
    )
    args = parser.parse_args()
    main(args)
