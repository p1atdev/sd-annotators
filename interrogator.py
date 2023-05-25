import torch
from typing import Optional, List, Tuple, Dict
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
from PIL import Image
from clip_interrogator import Config, Interrogator


HF_TOKEN = ""

CLIP_MODEL_SD_V1 = "ViT-L-14/openai"
CLIP_MODEL_SD_V2 = "ViT-H-14/laion2b_s32b_b79k"


DEFAULT_FILTER_TAGS = ["blurry", "meme", "parody"]
NSFW_TAGS = [
    "nude",
    "nipples",
    "pussy",
    "sex",
    "hetero",
    "penis",
]


class CustomInterrogator(Interrogator):
    def load_caption_model(self):
        print("nothing to do")
        pass


def check_tags(g_tags: Dict[str, float], filter_tags: List[str], threshold: float):
    for tag in filter_tags:
        rate = g_tags.get(tag, 0)
        if rate > threshold:
            return True
    return False


def save_caption(
    chunk,
    captions_dir,
    ci,
    filter_tags: Optional[List[str]],
    filtered_dir: Optional[Path],
    tag_options,
    extension: str,
    progress_bar,
):
    for image_path in chunk:
        if (captions_dir / (image_path.stem + ".txt")).exists():
            # print("Caption already exists, skipping")
            progress_bar.update(1)
            continue

        image = Image.open(image_path).convert("RGB")

        image_features = ci.image_to_features(image)

        medium = ci.mediums.rank(image_features, 1)[0]
        artist = ci.artists.rank(image_features, 1)[0]
        trending = ci.trendings.rank(image_features, 1)[0]
        movement = ci.movements.rank(image_features, 1)[0]
        flaves = ", ".join(ci.flavors.rank(image_features, 3))

        tags = []
        tags.append(medium)
        if tag_options["artist"]:
            tags.append(artist)
        if tag_options["trending"]:
            tags.append(trending)
        tags.append(movement)
        tags.append(flaves)

        # if is_sensitive:
        # for n_tag in NSFW_TAGS:
        #     if n_tag in tags:
        #         image_caption.insert(0, "nsfw")
        #         break

        concat = ", ".join(tags)

        if filter_tags is not None and filtered_dir is not None:
            for target_tag in filter_tags:
                if target_tag in tags:
                    os.rename(image_path, filtered_dir / image_path.name)
                    caption_path = filtered_dir / (image_path.stem + ".txt")
                    with open(caption_path, "w") as f:
                        f.write(concat)

                    print(f"Filter matched: {image_path}")
                    progress_bar.update(1)
                    continue

        caption_path = captions_dir / (image_path.stem + "." + extension)
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(concat)

        progress_bar.update(1)


def main(args):
    input_images_dir = Path(args.input_images_dir)
    captions_dir = args.captions_dir
    if captions_dir is None:
        captions_dir = input_images_dir
    else:
        captions_dir = Path(captions_dir)
    batch_size = args.threads
    model = args.model
    token = args.token
    filter_tags = args.filter_tags
    filtered_dir = args.filtered_dir
    tag_options = {
        "artist": args.artist,
        "trending": args.trending,
    }
    extension = args.extension

    if model == "v1":
        config = Config(clip_model_name=CLIP_MODEL_SD_V1)
    elif model == "v2":
        config = Config(clip_model_name=CLIP_MODEL_SD_V2)
    else:
        raise ValueError("Invalid model")

    if filtered_dir is not None:
        filtered_dir = Path(filtered_dir)
        filtered_dir.mkdir(exist_ok=True)
        if filter_tags is None:
            filter_tags = DEFAULT_FILTER_TAGS
            print(f"Filter tags not specified, using default: {filter_tags}")

    # png, jpg, jpeg, webp
    cropped_images = (
        list(input_images_dir.glob("*.png"))
        + list(input_images_dir.glob("*.jpg"))
        + list(input_images_dir.glob("*.jpeg"))
        + list(input_images_dir.glob("*.webp"))
    )

    print(f"Found {len(cropped_images)} images")

    chunks = np.array_split(cropped_images, batch_size)

    with tqdm(total=len(cropped_images)) as pbar:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for chunk in chunks:
                ci = CustomInterrogator(config)

                futures.append(
                    executor.submit(
                        save_caption,
                        chunk,
                        captions_dir,
                        ci,
                        filter_tags,
                        filtered_dir,
                        tag_options,
                        extension,
                        pbar,
                    )
                )

            for future in futures:
                future.result()

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_images_dir",
        type=str,
        help="Path to directory containing cropped images",
    )
    parser.add_argument(
        "--captions_dir",
        "-o",
        type=str,
        help="Path to directory to save captions",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=4,
        help="Number of threads to process images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Model to use for inference",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token for private repos",
    )
    parser.add_argument(
        "--filter_tags",
        type=str,
        nargs="+",
        default=None,
        help="Tags to filter",
    )
    parser.add_argument(
        "--filtered_dir",
        type=str,
        default=None,
        help="Path to directory to save filtered images",
    )
    parser.add_argument(
        "--extension",
        "-e",
        type=str,
        default="ci",
        help="Extension of caption file",
    )
    parser.add_argument(
        "--artist",
        action="store_true",
        help="Use artist tags",
    )
    parser.add_argument(
        "--trending",
        action="store_true",
        help="Use trending tags",
    )
    args = parser.parse_args()

    main(args)
