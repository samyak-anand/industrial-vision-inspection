"""Notebook cells 10, 13-14: exact SHA and cross-split pHash comparison.

This is exploratory analysis. pHash proximity does not establish physical-unit identity.
"""
from collections import defaultdict
from itertools import combinations
import hashlib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def iter_images(root):
    for split in ("train", "val", "test"):
        folder = root / "images" / split
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield split, path


PAIR_COLUMNS = ['split_1', 'image_1', 'path_1', 'split_2', 'image_2', 'path_2', 'phash_distance']


def exact_cross_split_duplicates(root):
    groups = defaultdict(set)
    for split, path in iter_images(root):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        groups[digest.hexdigest()].add(split)
    crossed = {digest: splits for digest, splits in groups.items() if len(splits) > 1}
    print('Unique SHA-256 hashes:', len(groups))
    print('Cross-split exact duplicate hashes:', len(crossed))
    return crossed


def compute_phashes(root):
    import imagehash
    from PIL import Image
    records = []
    for split, path in iter_images(root):
        with Image.open(path) as image:
            digest = str(imagehash.phash(image.convert('RGB')))
        records.append({'split': split, 'image_name': path.name, 'image_path': str(path), 'phash': digest})
    frame = pd.DataFrame(records, columns=['split', 'image_name', 'image_path', 'phash'])
    print('pHash images:', len(frame))
    print(frame['split'].value_counts().to_string())
    return frame


def find_cross_split_pairs(phash_df, threshold=10):
    import imagehash
    records = phash_df.to_dict('records')
    for record in records:
        record['hash_obj'] = imagehash.hex_to_hash(record['phash'])
    pairs = []
    for first, second in combinations(records, 2):
        if first['split'] == second['split']:
            continue
        distance = first['hash_obj'] - second['hash_obj']
        if distance <= threshold:
            pairs.append({'split_1': first['split'], 'image_1': first['image_name'],
                          'path_1': first['image_path'], 'split_2': second['split'],
                          'image_2': second['image_name'], 'path_2': second['image_path'],
                          'phash_distance': int(distance)})
    result = pd.DataFrame(pairs, columns=PAIR_COLUMNS)
    if not result.empty:
        result = result.sort_values('phash_distance').reset_index(drop=True)
    print(f'Cross-split pHash pairs (distance <= {threshold}):', len(result))
    return result

import math
import matplotlib.pyplot as plt
from PIL import Image
"""Notebook cells 16-18: map SHA-named exports to original filenames and inspect gaps."""



def load_metadata(project):
    path = project / 'data' / 'image_metadata.csv'
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {'image_id', 'image_name', 'image_path', 'sha256'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing metadata columns: {sorted(required - set(df.columns))}')
    print('Metadata records:', len(df))
    print('Columns:', df.columns.tolist())
    return df


def annotate_original_names(pairs, metadata):
    if 'sha256' not in metadata.columns:
        raise ValueError('image_metadata.csv requires sha256 for exported-name mapping')
    lookup = metadata.assign(export_stem=metadata['sha256'].astype(str).str[:16]).drop_duplicates('export_stem').set_index('export_stem')['image_name'].to_dict()
    result = pairs.copy()
    for side in (1, 2):
        result[f'original_{side}'] = result[f'image_{side}'].map(
            lambda filename: lookup.get(Path(filename).stem, 'NOT FOUND'))
    return result


def filename_gap_summary(annotated_pairs):
    result = annotated_pairs.copy()
    for side in (1, 2):
        result[f'number_{side}'] = pd.to_numeric(
            result[f'original_{side}'].map(lambda name: Path(str(name)).stem), errors='coerce')
    result['filename_gap'] = (result['number_1'] - result['number_2']).abs()
    print('Suspicious pairs:', len(result))
    for gap in (2, 5, 10):
        print(f'Filename gap <= {gap}:', int(result['filename_gap'].le(gap).sum()))
    if not result.empty:
        print(result['filename_gap'].describe().to_string())
    return result

"""Notebook cells 15, 19: optional similar-pair and source-sequence figures."""



def show_similar_pairs(pairs, count=5, save_to=None):
    top = pairs.nsmallest(count, 'phash_distance')
    if top.empty:
        print('No similar pairs to display.')
        return
    fig, axes = plt.subplots(len(top), 2, figsize=(12, 4 * len(top)), squeeze=False)
    for index, (_, pair) in enumerate(top.iterrows()):
        for col, side in enumerate((1, 2)):
            with Image.open(pair[f'path_{side}']) as im:
                axes[index, col].imshow(im.convert('RGB'))
            axes[index, col].set_title(f"{pair[f'split_{side}']}: {pair[f'image_{side}']}")
            axes[index, col].axis('off')
        axes[index, 1].set_xlabel(f"pHash distance: {pair['phash_distance']}")
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=130)
        print('Saved:', save_to)
    plt.close(fig)


def show_source_sequence(metadata, start=90, end=100, save_to=None):
    records = metadata.copy()
    records['sequence_number'] = records['image_name'].map(
        lambda name: int(Path(name).stem) if Path(name).stem.isdigit() else None)
    records = records[records['sequence_number'].between(start, end)].sort_values('sequence_number')
    if records.empty:
        print('No original filenames in selected range.')
        return
    columns = 4
    rows = math.ceil(len(records) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, 3.3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, (_, record) in zip(axes.flat, records.iterrows()):
        path = Path(record['image_path'])
        if not path.is_file():
            print('Missing source image:', path)
            continue
        with Image.open(path) as image:
            ax.imshow(image.convert('RGB'))
        ax.set_title(record['image_name'])
    fig.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=130)
        print('Saved:', save_to)
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Notebook similarity analysis (read-only)")
    parser.add_argument("--dataset", default="yolo_screw_hole")
    parser.add_argument("--threshold", type=int, default=10)
    args = parser.parse_args()
    root = PROJECT_ROOT / "data" / "processed" / args.dataset
    exact_cross_split_duplicates(root)
    hashes = compute_phashes(root)
    pairs = find_cross_split_pairs(hashes, threshold=args.threshold)
    if not pairs.empty:
        show_similar_pairs(pairs, save_to=PROJECT_ROOT / "reports" / "similarity" / "top_pairs.png")
    print("Analysis complete; no dataset files changed.")

if __name__ == "__main__":
    main()
