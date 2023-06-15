from PIL import Image
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import utils


def trim_images(image_files, output_folder, progress_bar):
    for image_file in image_files:
        img = Image.open(image_file)
        width, height = img.size

        if height > width:  # 縦長画像
            crop_height = int(height * 0.96)
        else:  # 横長画像
            crop_height = int(height * 0.90)

        cropped_img = img.crop((0, 0, width, crop_height))
        output_path = Path(output_folder) / Path(image_file).name
        cropped_img.save(output_path)

        progress_bar.update(1)


def main(input_folder, output_folder, batch_size):
    image_files = utils.glob_all_images(input_folder)

    os.makedirs(output_folder, exist_ok=True)

    chunks = np.array_split(image_files, batch_size)

    with tqdm(total=len(image_files)) as progress_bar:
        with ThreadPoolExecutor() as executor:
            features = []
            for chunk in chunks:
                executor.submit(trim_images, chunk, output_folder, progress_bar)

            for future in features:
                future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Input folder")
    parser.add_argument("output_folder", help="Output folder")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.batch_size)
