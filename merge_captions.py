import argparse
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def process_files(files, input, suffix, delete, pbar):
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            caption_prefix = f.read()

        file_name = file.stem

        # get suffix
        suffix_file = input / (file_name + "." + suffix)

        with open(suffix_file, "r", encoding="utf-8") as f:
            caption_suffix = f.read()

        # append prefix
        caption = caption_prefix + ", " + caption_suffix

        with open(file, "w", encoding="utf-8") as f:
            f.write(caption)

        if delete:
            os.remove(suffix_file)

        pbar.update(1)


def __main__(input, prefix, suffix, delete, threads):
    # get all files in input
    prefix_files = [
        p for p in input.iterdir() if (p.is_file and p.suffix.lower() == "." + prefix)
    ]

    chunks = np.array_split(prefix_files, threads)

    with tqdm(total=len(prefix_files)) as pbar:
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for chunk in chunks:
                futures.append(
                    executor.submit(process_files, chunk, input, suffix, delete, pbar)
                )

            for future in futures:
                future.result()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Input directory")
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="Prefix to append caption file extension",
        default="txt",
    )
    parser.add_argument(
        "--suffix",
        "-s",
        type=str,
        help="Suffix to append caption file extension",
        default="caption",
    )
    parser.add_argument("--delete", help="Delete suffix file", action="store_true")
    parser.add_argument(
        "--threads", "-t", type=int, help="Number of threads", default=100
    )

    args = parser.parse_args()

    __main__(Path(args.input), args.prefix, args.suffix, args.delete, args.threads)
