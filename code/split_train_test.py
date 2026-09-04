"""Train/test split for cross-concentration generalization (Table S4).

8:2 split with a segment-level holdout: each gas's 100 samples are divided
into 10-sample segments (one concentration each), and 2 of the 10 segments
per gas are held out for testing. AIR uses a random split instead. Supports
multiple seeds to generate different split schemes.
"""
import argparse
import os
import random

import numpy as np
from sklearn.model_selection import train_test_split

CLASS_NAMES = ["H2S", "CH4", "CO2", "AIR"]


def load_data(data_dir):
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    print(f"data: X={X.shape}, y={y.shape}, "
          f"counts={dict(zip(CLASS_NAMES, np.bincount(y)))}")
    return X, y


def split_by_segments(X, y, test_ratio=0.2, air_strategy="random_split",
                       random_state=42, segment_sizes=None,
                       use_random_sampling=False):
    """Segment-level (concentration-level) train/test split."""
    if segment_sizes is None:
        segment_sizes = {}
    default_segment_size = 10
    train_indices, test_indices = [], []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        idx = np.where(y == class_idx)[0]
        n = len(idx)

        # AIR: random split (single acquisition, cannot segment)
        if class_name == "AIR":
            if air_strategy == "all_train":
                train_indices.extend(idx.tolist())
            else:
                tr, te = train_test_split(idx, test_size=test_ratio,
                                            random_state=random_state)
                train_indices.extend(tr.tolist())
                test_indices.extend(te.tolist())
            continue

        if use_random_sampling:
            tr, te = train_test_split(idx, test_size=test_ratio,
                                        random_state=random_state, shuffle=True)
            train_indices.extend(tr.tolist())
            test_indices.extend(te.tolist())
        else:
            seg = segment_sizes.get(class_name, default_segment_size)
            n_seg = n // seg
            n_test_seg = max(1, int(n_seg * test_ratio))
            all_ids = list(range(n_seg))
            test_ids = random.sample(all_ids, n_test_seg)
            train_ids = [s for s in all_ids if s not in test_ids]
            for s in train_ids:
                train_indices.extend(idx[s * seg:(s + 1) * seg].tolist())
            for s in test_ids:
                test_indices.extend(idx[s * seg:(s + 1) * seg].tolist())
            # remainder -> train
            rem = n_seg * seg
            if rem < n:
                train_indices.extend(idx[rem:].tolist())

    train_indices = np.array(train_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)
    print(f"train: {len(train_indices)} ({len(train_indices)/len(y)*100:.0f}%), "
          f"test: {len(test_indices)} ({len(test_indices)/len(y)*100:.0f}%)")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c}: train={int((y[train_indices]==i).sum())}, "
              f"test={int((y[test_indices]==i).sum())}")
    return train_indices, test_indices


def save_split(X, y, train_idx, test_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    np.save(os.path.join(output_dir, "train_indices.npy"), train_idx)
    np.save(os.path.join(output_dir, "test_indices.npy"), test_idx)
    print(f"-> {output_dir}/  (X_train {X_train.shape}, X_test {X_test.shape})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--data_dir", type=str, default="data/preprocessed")
    p.add_argument("--output_dir", type=str, default="data/concentration_split")
    p.add_argument("--air_strategy", type=str, default="random_split",
                    choices=["random_split", "all_train"])
    p.add_argument("--segment_sizes", type=str, default=None,
                    help="per-gas segment size, e.g. 'H2S:10,CH4:10,CO2:10'")
    p.add_argument("--random_sampling", action="store_true",
                    help="random sampling instead of contiguous segments")
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    seg_sizes = None
    if args.segment_sizes:
        seg_sizes = {}
        for item in args.segment_sizes.split(","):
            if ":" in item:
                g, s = item.split(":")
                seg_sizes[g.strip().upper()] = int(s.strip())

    X, y = load_data(args.data_dir)
    tr, te = split_by_segments(X, y, test_ratio=args.test_ratio,
                                air_strategy=args.air_strategy,
                                random_state=args.seed,
                                segment_sizes=seg_sizes,
                                use_random_sampling=args.random_sampling)
    save_split(X, y, tr, te, args.output_dir)


if __name__ == "__main__":
    main()
