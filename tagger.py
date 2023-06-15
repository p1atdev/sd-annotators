# Ref: https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py
# thanks to smilingwolf

import torch
from typing import Optional, List, Tuple, Dict
from PIL import Image
import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
import cv2
import gc
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import os
import utils

HF_TOKEN = ""

MOAT_MODEL_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
SWIN_MODEL_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
# CONV_MODEL_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

DEFAULT_FILTER_TAGS = ["blurry", "meme", "parody"]
NSFW_TAGS = [
    "nude",
    "nipples",
    "pussy",
    "sex",
    "hetero",
    "penis",
]


def smart_imread(img, flag=cv2.IMREAD_UNCHANGED):
    if img.endswith(".gif"):
        img = Image.open(img)
        img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(img, flag)
    return img


def smart_24bit(img):
    if img.dtype is np.dtype(np.uint16):
        img = (img / 257).astype(np.uint8)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        trans_mask = img[:, :, 3] == 0
        img[trans_mask] = [255, 255, 255, 255]
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return new_im


def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


class Tagger:
    def __init__(
        self, model_repo: str, model_filename: str, hf_token: Optional[str] = None
    ) -> rt.InferenceSession:
        if hf_token is None:
            hf_token = HF_TOKEN
        model_path = huggingface_hub.hf_hub_download(
            model_repo, model_filename, use_auth_token=hf_token
        )
        self.model = rt.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        csv_path = huggingface_hub.hf_hub_download(
            CONV2_MODEL_REPO, LABEL_FILENAME, use_auth_token=HF_TOKEN
        )
        df = pd.read_csv(csv_path)

        self.tag_names = df["name"].tolist()
        self.rating_indexes = list(np.where(df["category"] == 9)[0])
        self.general_indexes = list(np.where(df["category"] == 0)[0])
        self.character_indexes = list(np.where(df["category"] == 4)[0])

    def predict(
        self, _image: Image.Image, general_threshold: float, character_threshold: float
    ):
        model = self.model

        image = _image.copy()

        _, height, width, _ = model.get_inputs()[0].shape

        # Alpha to white
        image = image.convert("RGBA")
        new_image = Image.new("RGBA", image.size, "WHITE")
        new_image.paste(image, mask=image)
        image = new_image.convert("RGB")
        image = np.asarray(image)

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = make_square(image, height)
        image = smart_resize(image, height)
        image = image.astype(np.float32)
        image = np.expand_dims(image, 0)

        input_name = model.get_inputs()[0].name
        label_name = model.get_outputs()[0].name
        probs = model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, probs[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]
        general_res = [x for x in general_names if x[1] > general_threshold]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]
        character_res = [x for x in character_names if x[1] > character_threshold]
        character_res = dict(character_res)

        b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
        tags = (
            ", ".join(list(b.keys())).replace("_", " ")
            # .replace("(", "(")
            # .replace(")", ")")
        )
        original = ", ".join(list(b.keys()))

        return tags, original, rating, character_res, general_res

    def unload(self):
        del self.model
        torch.cuda.empty_cache()
        gc.collect()


def check_tags(g_tags: Dict[str, float], filter_tags: List[str], threshold: float):
    for tag in filter_tags:
        rate = g_tags.get(tag, 0)
        if rate > threshold:
            return True
    return False


def save_caption(
    chunk,
    captions_dir,
    tagger,
    filter_tags: Optional[List[str]],
    filtered_dir: Optional[Path],
    filter_threshold: Optional[float],
    caption_ext: str,
    progress_bar,
):
    for image_path in chunk:
        if (captions_dir / (image_path.stem + f".{caption_ext}")).exists():
            # print("Caption already exists, skipping")
            progress_bar.update(1)
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except:
            raise ValueError(f"Failed to open image: {image_path}")

        tags, _, rating, _, original = tagger.predict(image, 0.35, 0.8)

        # is_sensitive = rating["sensitive"] > 0.9

        image_caption = []
        # if is_sensitive:
        for n_tag in NSFW_TAGS:
            if n_tag in tags:
                image_caption.insert(0, "nsfw")
                break
        # image_caption.append(caption[0])
        image_caption.append(tags)

        concat = ", ".join(image_caption)

        # print(concat)
        # print(caption_path)

        if (
            filter_tags is not None
            and filtered_dir is not None
            and filter_threshold is not None
        ):
            if check_tags(original, filter_tags, filter_threshold):
                os.rename(image_path, filtered_dir / image_path.name)
                caption_path = filtered_dir / (image_path.stem + f".{caption_ext}")
                with open(caption_path, "w") as f:
                    f.write(concat)

                print(f"Filter matched: {image_path}")
                progress_bar.update(1)
                continue

        caption_path = captions_dir / (image_path.stem + f".{caption_ext}")
        with open(caption_path, "w") as f:
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
    filter_threshold = args.filter_threshold
    caption_ext = args.extension

    if model == "vit":
        model = VIT_MODEL_REPO
    elif model == "conv2":
        model = CONV2_MODEL_REPO
    elif model == "swinv2":
        model = SWIN_MODEL_REPO
    elif model == "moat":
        model = MOAT_MODEL_REPO
    else:
        raise ValueError("Invalid model")

    if filtered_dir is not None:
        filtered_dir = Path(filtered_dir)
        filtered_dir.mkdir(exist_ok=True)
        if filter_tags is None:
            filter_tags = DEFAULT_FILTER_TAGS
            print(f"Filter tags not specified, using default: {filter_tags}")
        if filter_threshold is None:
            filter_threshold = 0.8
            print(f"Filter threshold not specified, using default: {filter_threshold}")

    # png, jpg, jpeg, webp
    cropped_images = utils.glob_all_images(input_images_dir)

    print(f"Found {len(cropped_images)} images")

    chunks = np.array_split(cropped_images, batch_size)

    with tqdm(total=len(cropped_images)) as pbar:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for chunk in chunks:
                tagger = Tagger(VIT_MODEL_REPO, MODEL_FILENAME, token)

                futures.append(
                    executor.submit(
                        save_caption,
                        chunk,
                        captions_dir,
                        tagger,
                        filter_tags,
                        filtered_dir,
                        filter_threshold,
                        caption_ext,
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
        default=10,
        help="Number of threads to process images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit",
        choices=["vit", "conv2", "swinv2", "moat"],
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
        "--filter_threshold",
        type=float,
        default=None,
        help="Filter threshold for tags",
    )
    parser.add_argument(
        "--extension",
        "-e",
        type=str,
        default="txt",
        help="Extension for captions",
    )
    args = parser.parse_args()

    main(args)
