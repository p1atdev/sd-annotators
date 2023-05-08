import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import numpy as np
import gc
from tqdm import tqdm
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor


class BLIP2:
    def __init__(self, model_type="coco"):
        # setup device to use
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2", model_type=model_type, is_eval=True, device=self.device
        )

    def generate_caption(
        self,
        _image: Image,
        num_beams: int = 3,  # number of beams to use for beam search. 1 means no beam search.
        use_nucleus_sampling: bool = False,  # if False, use top-k sampling
        max_length: int = 30,  # maximum length of the generated caption
        min_length: int = 10,  # minimum length of the generated caption
        top_p: float = 0.9,  # The cumulative probability for nucleus sampling.
        repetition_penalty: float = 0.9,  # The parameter for repetition penalty. 1.0 means no penalty.
    ):
        # loads BLIP-2 pre-trained model
        # prepare the image
        image = self.vis_processors["eval"](_image).unsqueeze(0).to(self.device)

        captions = self.model.generate(
            {"image": image},
            num_beams=num_beams,
            use_nucleus_sampling=use_nucleus_sampling,
            max_length=max_length,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        return captions

    def unload(self):
        del self.model
        del self.vis_processors
        torch.cuda.empty_cache()
        gc.collect()


def replacer(caption: str):
    to_girl = ["woman", "anime character", "anime girl"]
    for word in to_girl:
        caption = caption.replace(word, "girl")

    return caption.strip()


def process_images(images, output, ext, overwrite, progress_bar):
    blip2 = BLIP2()

    for image_path in images:
        # print(f"Processing {image_path}")
        caption_path = output / (image_path.stem + f".{ext}")

        if caption_path.exists() and not overwrite:
            progress_bar.update(1)
            continue

        image = Image.open(image_path).convert("RGB")

        caption = blip2.generate_caption(image, use_nucleus_sampling=True)[0]
        caption = replacer(caption)

        # save caption
        with open(caption_path, "w") as f:
            f.write(caption)

        progress_bar.update(1)


def main(args):
    input_images_dir = Path(args.input_images_dir)
    captions_dir = args.captions_dir
    if captions_dir is None:
        captions_dir = input_images_dir
    else:
        captions_dir = Path(captions_dir)
    threads = args.threads
    model = args.model
    ext = args.ext
    overwrite = args.overwrite

    # png, jpg, jpeg, webp
    cropped_images = (
        list(input_images_dir.glob("*.png"))
        + list(input_images_dir.glob("*.jpg"))
        + list(input_images_dir.glob("*.jpeg"))
        + list(input_images_dir.glob("*.webp"))
    )

    print(f"Found {len(cropped_images)} images")

    chunks = np.array_split(cropped_images, threads)

    with tqdm(total=len(cropped_images)) as pbar:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for chunk in chunks:
                futures.append(
                    executor.submit(
                        process_images, chunk, captions_dir, ext, overwrite, pbar
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
        default=1,
        help="Number of threads to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="coco",
        choices=["coco"],
        help="Model to use for inference",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="caption",
        choices=["txt", "caption"],
        help="Extension to use for saving captions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing captions",
    )
    args = parser.parse_args()

    main(args)
