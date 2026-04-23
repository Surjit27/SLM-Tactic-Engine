"""Generate random synthetic dataset aligned to obs[256], action[32]."""

from __future__ import annotations

import argparse
import os

import config
from data_utils import generate_split, save_split


def generate_and_save(train_samples: int, val_samples: int, seed: int) -> tuple[str, str]:
    train = generate_split(train_samples, seed)
    val = generate_split(val_samples, seed + 1)

    os.makedirs(config.DATA_DIR, exist_ok=True)
    train_path = os.path.join(config.DATA_DIR, "train.npz")
    val_path = os.path.join(config.DATA_DIR, "val.npz")
    save_split(train_path, train)
    save_split(val_path, val)

    print(f"Saved train split: {train_path} ({train_samples} samples)")
    print(f"Saved val split:   {val_path} ({val_samples} samples)")
    return train_path, val_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Farcana-style dataset")
    parser.add_argument("--train-samples", type=int, default=config.DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=config.DEFAULT_VAL_SAMPLES)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    generate_and_save(args.train_samples, args.val_samples, args.seed)


if __name__ == "__main__":
    main()
