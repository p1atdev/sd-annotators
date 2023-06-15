from PIL import Image
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import utils


def process_images(images, output_directory, progress_bar):
    # ディレクトリ内の画像を回す
    for file_path in images:
        stem = Path(file_path).stem

        if (output_directory / f"{stem} (1).png").exists():
            progress_bar.update(1)
            continue

        # print("file_path: ", file_path)

        im = Image.open(file_path)

        # 画像を四分割する
        width, height = im.size
        left_upper = im.crop((0, 0, width / 2, height / 2))
        right_upper = im.crop((width / 2, 0, width, height / 2))
        left_lower = im.crop((0, height / 2, width / 2, height))
        right_lower = im.crop((width / 2, height / 2, width, height))

        # 四分割した画像を保存する
        left_upper.save(output_directory / f"{stem} (1).png")
        right_upper.save(output_directory / f"{stem} (2).png")
        left_lower.save(output_directory / f"{stem}(3).png")
        right_lower.save(output_directory / f"{stem} (4).png")

        progress_bar.update(1)


def __main__(input, output, threads):
    input_directory = Path(input).resolve()
    output_directory = Path(output).resolve()

    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    images = utils.glob_all_images(input_directory)

    chunks = np.array_split(images, threads)

    with tqdm(total=len(images)) as progress_bar:
        with ThreadPoolExecutor() as executor:
            features = []
            for chunk in chunks:
                features.append(
                    executor.submit(
                        process_images, chunk, output_directory, progress_bar
                    )
                )

            for feature in features:
                feature.result()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--threads", type=int, default=20)
    args = parser.parse_args()
    __main__(args.input, args.output, args.threads)
