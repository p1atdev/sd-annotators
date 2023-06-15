import os
from pathlib import Path
import torch
from transformers import pipeline
from typing import Union, Literal, Optional, Dict
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
import utils

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_AESTHETIC = "cafeai/cafe_aesthetic"
MODEL_STYLE = "cafeai/cafe_style"
MODEL_WAIFU = "cafeai/cafe_waifu"

MODEL_MAP = {
    "aesthetic": {
        "model": MODEL_AESTHETIC,
        "top_k": 2,
        "folders": ["aesthetic", "not_aesthetic", "excluded"],
    },
    "style": {
        "model": MODEL_STYLE,
        "top_k": 5,
        "folders": ["anime", "real_life", "3d", "manga like", "other", "excluded"],
    },
    "waifu": {
        "model": MODEL_WAIFU,
        "top_k": 5,
        "folders": ["waifu", "not_waifu", "excluded"],
    },
}


class Predictor:
    def __init__(
        self,
        model_type: Union[Literal["aesthetic"], Literal["style"], Literal["waifu"]],
    ) -> None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_type = MODEL_MAP[model_type]

        self.model = pipeline(
            "image-classification",
            model=self.model_type["model"],
            device=device,
        )

    def predict(self, image: Image) -> Dict[str, float]:
        prediction = self.model(image, top_k=self.model_type["top_k"])
        result = {}
        for pred in prediction:
            result[pred["label"]] = pred["score"]
        return result


def search_same_stem_file(path: Path, file: Path):
    return list(path.glob(f"{file.stem}.*"))


def process_images(
    images, output, model: Predictor, threshold: Optional[float], progress_bar
):
    for image in images:
        files = search_same_stem_file(output, image)
        with Image.open(image) as img:
            prediction = model.predict(img)
            if threshold:
                done = False
                for label, score in prediction.items():
                    if score > threshold:
                        for file in files:
                            file.rename(output / label / file.name)
                        done = True
                        break

                if not done:
                    # move to excluded
                    for file in files:
                        file.rename(output / "excluded" / file.name)

            else:
                # get highest one
                data = sorted(prediction.items(), key=lambda x: x[1], reverse=True)[0]
                label = data[0]
                for file in files:
                    file.rename(output / label / file.name)

        progress_bar.update(1)


def main(args):
    input_path = Path(args.input_path)
    output_path = args.output_path
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
    batch_size = args.batch_size
    threshold = args.threshold
    model_type = args.model_type

    if not output_path.exists():
        output_path.mkdir(parents=True)

    for folder in MODEL_MAP[model_type]["folders"]:
        (output_path / folder).mkdir(parents=True, exist_ok=True)

    images = utils.glob_all_images(input_path)

    print(f"Found {len(images)} images")

    chunks = np.array_split(images, batch_size)

    with tqdm(total=len(images)) as pbar:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = []
            for chunk in chunks:
                model = Predictor(model_type)
                futures.append(
                    executor.submit(
                        process_images, chunk, output_path, model, threshold, pbar
                    )
                )

            for future in futures:
                future.result()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify images based on aesthetic, style, or waifu"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input images",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Path to output images",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=10,
        help="Batch size",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help="Threshold for classification",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default="aesthetic",
        choices=["aesthetic", "style", "waifu"],
        help="Model type",
    )
    args = parser.parse_args()
    main(args)
