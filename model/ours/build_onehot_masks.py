from pathlib import Path
import numpy as np
from PIL import Image
import argparse


def convert_dir(src_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(list(src_dir.glob('*.png')) + list(src_dir.glob('*.PNG')))
    if not files:
        print(f'[Skip] No PNG masks found in {src_dir}.')
        return 0

    for fp in files:
        arr = np.array(Image.open(fp))
        if arr.ndim == 3:
            arr = arr[..., 0]

        # Class order: 0=background, 1=intact concrete, 2=spalled concrete
        bg = (arr == 0).astype(np.uint8) * 255
        concrete = (arr == 1).astype(np.uint8) * 255
        spall = (arr == 2).astype(np.uint8) * 255

        onehot = np.stack([bg, concrete, spall], axis=-1)
        Image.fromarray(onehot, mode='RGB').save(out_dir / fp.name)

    print(f'[Done] {src_dir} -> {out_dir}, {len(files)} files converted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert label masks to three-channel one-hot masks.'
    )
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    n_files = convert_dir(args.input_dir, args.output_dir)
    print(f'[Summary] {n_files} masks converted.')
