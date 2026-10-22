#baseline training 

from pathlib import Path
import sys
import json
import shutil
import hashlib
import random

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from IPython.display import display

PROJECT_ROOT = Path(
    r"C:\Users\samya\PyCharmProject\industrial-vision-inspection"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DATASET_DIR,
    DATA_DIR,
    IMAGE_METADATA_FILE,
    OBJECT_ANNOTATIONS_FILE,
)

DATASET_DIR = Path(DATASET_DIR)
DATA_DIR = Path(DATA_DIR)

METADATA_PATH = Path(IMAGE_METADATA_FILE)
ANNOTATIONS_PATH = Path(OBJECT_ANNOTATIONS_FILE)

OUTPUT_DIR = DATA_DIR / "processed" / "yolo_screw_hole"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
REPORT_DIR = OUTPUT_DIR / "reports"


for directory in [OUTPUT_DIR,MANIFEST_DIR,REPORT_DIR]:
    directory.mkdir(parents=True,exist_ok=True)

RANDOM_SEED= 42
TRAIN_RATIO=0.70
VAL_RATIO=0.15
TEST_RATIO=0.15

CLASS_NAME=["screw","hole"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAME)}

# Configure this after inspecting the actual review_status values.
# Example: {"rejected", "needs_review"}
EXCLUDED_REVIEW_STATUSES = set()

print("Project:", PROJECT_ROOT)
print("Raw images:", DATASET_DIR)
print("Output:", OUTPUT_DIR)
print("Classes:", CLASS_TO_ID)

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9
assert DATASET_DIR.is_dir(), "Dataset directory not found."
assert METADATA_PATH.is_file(), f"Metadata CSV not found: {METADATA_PATH}"
assert ANNOTATIONS_PATH.is_file(), (
    f"Annotation CSV not found: {ANNOTATIONS_PATH}"
)
metadata_df = pd.read_csv(METADATA_PATH)
annotations_df= pd.read_csv(ANNOTATIONS_PATH)

required_metadata={
    'image_id','image_path','image_width_px','image_height_px'
}

required_annotations={
    "image_id", "label", "shape_type",
    "x1_px", "y1_px", "x2_px", "y2_px"
}

missing_metadata = required_metadata - set(metadata_df.columns)
missing_annotations = required_annotations - set(annotations_df.columns)

if missing_metadata:
    raise ValueError(f"Missing metadata columns: {sorted(missing_metadata)}")

if missing_annotations:
    raise ValueError(
        f"Missing annotation columns: {sorted(missing_annotations)}"
    )
print("Images in metadata:", len(metadata_df))
print("Annotation rows:", len(annotations_df))

print("\nLabels:")
display(annotations_df["label"].value_counts(dropna=False))

if "review_status" in annotations_df.columns:
    print("\nReview status:")
    display(annotations_df["review_status"].value_counts(dropna=False))

if "shape_type" in annotations_df.columns:
    print("\nShape types:")
    display(annotations_df["shape_type"].value_counts(dropna=False))

def resolve_image_path(value):
    """Resolve an image path from metadata without modifying the source."""
    if pd.isna(value):
        return None

    path = Path(str(value))

    candidates = [
        path,
        PROJECT_ROOT / path,
        DATASET_DIR / path.name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


image_records = []

for row in metadata_df.itertuples(index=False):
    record = {
        "image_id": str(row.image_id),
        "source_path": str(row.image_path),
        "resolved_path": None,
        "image_exists": False,
        "image_readable": False,
        "actual_width": None,
        "actual_height": None,
        "metadata_dimensions_match": False,
        "error": None,
    }

    image_path = resolve_image_path(row.image_path)

    if image_path is not None:
        record["resolved_path"] = str(image_path)
        record["image_exists"] = True

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                width, height = image.size

            record["image_readable"] = True
            record["actual_width"] = width
            record["actual_height"] = height
            record["metadata_dimensions_match"] = (
                width == int(row.image_width_px)
                and height == int(row.image_height_px)
            )

        except Exception as exc:
            record["error"] = str(exc)
    else:
        record["error"] = "Image file not found"

    image_records.append(record)

image_check_df = pd.DataFrame(image_records)

display(
    image_check_df[
        [
            "image_exists",
            "image_readable",
            "metadata_dimensions_match",
        ]
    ].sum().to_frame("passed_images")
)

display(image_check_df[image_check_df["error"].notna()].head(20))

def prepare_annotations(annotations, image_checks):
    """Validate annotations and generate normalized YOLO label rows."""
    image_lookup = image_checks.set_index("image_id").to_dict("index")

    valid_records = []
    rejected_records = []

    for index, row in annotations.iterrows():
        image_id = str(row["image_id"])
        label = str(row["label"]).strip().lower()
        shape_type = str(row["shape_type"]).strip().lower()

        def reject(reason):
            rejected_records.append({
                "source_row": index,
                "image_id": image_id,
                "label": label,
                "reason": reason,
            })

        if image_id not in image_lookup:
            reject("Image missing from metadata")
            continue

        image_info = image_lookup[image_id]

        if not (
            image_info["image_exists"]
            and image_info["image_readable"]
            and image_info["metadata_dimensions_match"]
        ):
            reject("Image failed file or dimension validation")
            continue

        if label not in CLASS_TO_ID:
            reject("Unknown or unsupported class")
            continue

        if shape_type != "rectangle":
            reject("Unsupported shape type")
            continue

        if "review_status" in annotations.columns:
            status = (
                ""
                if pd.isna(row["review_status"])
                else str(row["review_status"]).strip().lower()
            )

            if status in EXCLUDED_REVIEW_STATUSES:
                reject(f"Excluded review status: {status}")
                continue

        try:
            coords = np.array([
                row["x1_px"], row["y1_px"],
                row["x2_px"], row["y2_px"]
            ], dtype=float)

            if not np.isfinite(coords).all():
                raise ValueError("Non-finite coordinates")

            x1, y1, x2, y2 = coords

            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            width = x_max - x_min
            height = y_max - y_min

            if width <= 0 or height <= 0:
                raise ValueError("Zero or negative box dimensions")

            image_width = image_info["actual_width"]
            image_height = image_info["actual_height"]

            if (
                x_min < 0 or y_min < 0
                or x_max > image_width
                or y_max > image_height
            ):
                raise ValueError("Bounding box out of image bounds")

            x_center = ((x_min + x_max) / 2) / image_width
            y_center = ((y_min + y_max) / 2) / image_height
            norm_width = width / image_width
            norm_height = height / image_height

            normalized = [
                x_center, y_center, norm_width, norm_height
            ]

            if not np.isfinite(normalized).all():
                raise ValueError("Invalid normalized coordinates")

            if not all(0 <= value <= 1 for value in normalized):
                raise ValueError("Normalized coordinates outside [0, 1]")

            valid_records.append({
                "image_id": image_id,
                "class_name": label,
                "class_id": CLASS_TO_ID[label],
                "x_center": x_center,
                "y_center": y_center,
                "width": norm_width,
                "height": norm_height,
            })

        except (TypeError, ValueError, OverflowError) as exc:
            reject(str(exc))

    return pd.DataFrame(valid_records), pd.DataFrame(rejected_records)


valid_annotations, rejected_annotations = prepare_annotations(
    annotations_df, image_check_df
)

print("Valid annotation rows:", len(valid_annotations))
print("Rejected annotation rows:", len(rejected_annotations))

if not rejected_annotations.empty:
    display(rejected_annotations["reason"].value_counts().to_frame("count"))
    display(rejected_annotations.head(20))

display(valid_annotations.head())
valid_images = image_check_df[
    image_check_df["image_exists"]
    & image_check_df["image_readable"]
    & image_check_df["metadata_dimensions_match"]
].copy()

# Ensure one metadata row per image ID.
if valid_images["image_id"].duplicated().any():
    duplicates = valid_images[
        valid_images["image_id"].duplicated(keep=False)
    ]
    display(duplicates)
    raise ValueError("Duplicate image_id values need to be resolved.")

# Build image-level class information.
class_sets = (
    valid_annotations.groupby("image_id")["class_name"]
    .agg(lambda values: sorted(set(values)))
    .to_dict()
)

valid_images["classes"] = valid_images["image_id"].map(
    lambda image_id: class_sets.get(image_id, [])
)

valid_images["annotation_count_valid"] = (
    valid_annotations.groupby("image_id").size()
    .reindex(valid_images["image_id"])
    .fillna(0)
    .astype(int)
    .to_numpy()
)

# Preserve original image dimensions and source paths.
valid_images["group_id"] = valid_images["image_id"]

display(
    valid_images[
        ["image_id", "actual_width", "actual_height",
         "classes", "annotation_count_valid"]
    ].head()
)

print("Images passing file validation:", len(valid_images))
print("Images with valid annotations:", (valid_images["annotation_count_valid"] > 0).sum())
print("Images with zero valid annotations:", (valid_images["annotation_count_valid"] == 0).sum())
from sklearn.model_selection import GroupShuffleSplit

# Attach original hashes where available.
if "sha256" in metadata_df.columns:
    hash_lookup = metadata_df.set_index(
        metadata_df["image_id"].astype(str)
    )["sha256"].to_dict()

    valid_images["sha256"] = valid_images["image_id"].map(hash_lookup)

    valid_images["group_id"] = valid_images.apply(
        lambda row: (
            f"sha256:{row['sha256']}"
            if pd.notna(row["sha256"]) and str(row["sha256"]).strip()
            else f"image:{row['image_id']}"
        ),
        axis=1,
    )
else:
    valid_images["group_id"] = (
        "image:" + valid_images["image_id"].astype(str)
    )

print("Unique images:", len(valid_images))
print("Unique split groups:", valid_images["group_id"].nunique())

duplicate_groups = (
    valid_images.groupby("group_id")
    .size()
    .loc[lambda counts: counts > 1]
)

print("Groups with exact duplicate files:", len(duplicate_groups))
display(duplicate_groups.head(10).to_frame("image_count"))

# First reserve 15% for the final test set.
splitter_test = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_RATIO,
    random_state=RANDOM_SEED,
)

train_val_idx, test_idx = next(
    splitter_test.split(
        valid_images,
        groups=valid_images["group_id"]
    )
)

train_val = valid_images.iloc[train_val_idx].copy()
test_images = valid_images.iloc[test_idx].copy()

# Validation must be 15% of the entire dataset.
val_fraction_of_remaining = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)

splitter_val = GroupShuffleSplit(
    n_splits=1,
    test_size=val_fraction_of_remaining,
    random_state=RANDOM_SEED,
)

train_idx, val_idx = next(
    splitter_val.split(
        train_val,
        groups=train_val["group_id"]
    )
)

train_images = train_val.iloc[train_idx].copy()
val_images = train_val.iloc[val_idx].copy()

train_images["split"] = "train"
val_images["split"] = "val"
test_images["split"] = "test"

split_manifest = pd.concat(
    [train_images, val_images, test_images],
    ignore_index=True
)

print("Train:", len(train_images))
print("Validation:", len(val_images))
print("Test:", len(test_images))
print("Total:", len(split_manifest))

def count_classes_for_split(image_df, annotation_df):
    image_ids = set(image_df["image_id"].astype(str))

    subset = annotation_df[
        annotation_df["image_id"].astype(str).isin(image_ids)
    ]

    counts = subset["class_name"].value_counts()

    return {
        "images": len(image_df),
        "annotated_images": int(
            image_df["annotation_count_valid"].gt(0).sum()
        ),
        "valid_annotations": len(subset),
        **{
            f"{class_name}_annotations": int(counts.get(class_name, 0))
            for class_name in CLASS_NAME
        },
    }


split_distribution = pd.DataFrame({
    "train": count_classes_for_split(train_images, valid_annotations),
    "val": count_classes_for_split(val_images, valid_annotations),
    "test": count_classes_for_split(test_images, valid_annotations),
}).T

display(split_distribution)
# Add the original annotation count from metadata.
source_counts = metadata_df.set_index(
    metadata_df["image_id"].astype(str)
)["annotation_count"].to_dict() if "annotation_count" in metadata_df.columns else {}

valid_images["source_annotation_count"] = (
    valid_images["image_id"]
    .map(source_counts)
)

# Conservative eligibility rules:
# 1. Include images with at least one valid target annotation.
# 2. Include zero-annotation images only when the source explicitly
#    reports zero annotations.
# 3. Exclude images whose original annotations existed but all failed.

has_valid_labels = valid_images["annotation_count_valid"] > 0

verified_empty = (
    valid_images["source_annotation_count"].notna()
    & (valid_images["source_annotation_count"] == 0)
)

eligible_images = valid_images[
    has_valid_labels | verified_empty
].copy()

excluded_images = valid_images[
    ~(has_valid_labels | verified_empty)
].copy()

print("Eligible images:", len(eligible_images))
print("Excluded images requiring review:", len(excluded_images))

valid_images = eligible_images.copy()

display(
    excluded_images[
        ["image_id", "source_annotation_count", "annotation_count_valid"]
    ].head(20)
)
SPLITS = {
    "train": train_images,
    "val": val_images,
    "test": test_images,
}

for split in SPLITS:
    for category in ["images", "labels"]:
        path = OUTPUT_DIR / category / split
        path.mkdir(parents=True, exist_ok=True)

print("YOLO directories created.")

import re

# Map each source image ID to its validated file path.
path_lookup = image_check_df.set_index("image_id")["resolved_path"].to_dict()

# Group valid annotations by image.
annotations_by_image = {
    str(image_id): group
    for image_id, group in valid_annotations.groupby("image_id")
}

# Track generated filenames for traceability.
export_records = []
used_filenames = set()

for split_name, split_df in SPLITS.items():
    images_dir = OUTPUT_DIR / "images" / split_name
    labels_dir = OUTPUT_DIR / "labels" / split_name

    # Remove stale generated files from earlier runs.
    # This only affects the dedicated processed YOLO directories.
    for directory in [images_dir, labels_dir]:
        for old_file in directory.iterdir():
            if old_file.is_file():
                old_file.unlink()

    for row in split_df.itertuples(index=False):
        image_id = str(row.image_id)
        source_path = Path(path_lookup[image_id])

        # Use a filesystem-safe ID and retain the image extension.
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", image_id)
        output_name = f"{safe_id}{source_path.suffix.lower()}"

        if output_name in used_filenames:
            raise ValueError(
                f"Duplicate output filename: {output_name}"
            )
        used_filenames.add(output_name)

        destination_image = images_dir / output_name
        destination_label = labels_dir / f"{safe_id}.txt"

        shutil.copy2(source_path, destination_image)

        rows = annotations_by_image.get(image_id, [])

        with destination_label.open("w", encoding="utf-8") as file:
            for annotation in rows.itertuples(index=False):
                values = [
                    annotation.x_center,
                    annotation.y_center,
                    annotation.width,
                    annotation.height,
                ]

                file.write(
                    f"{annotation.class_id} "
                    + " ".join(f"{value:.6f}" for value in values)
                    + "\n"
                )

        export_records.append({
            "image_id": image_id,
            "split": split_name,
            "source_image": str(source_path),
            "exported_image": str(destination_image),
            "exported_label": str(destination_label),
            "annotation_count": len(rows),
            "empty_label_file": len(rows) == 0,
        })

export_manifest = pd.DataFrame(export_records)

display(export_manifest.groupby("split").agg(
    images=("image_id", "count"),
    annotations=("annotation_count", "sum"),
    empty_labels=("empty_label_file", "sum"),
))

data_yaml = {
    "path": OUTPUT_DIR.resolve().as_posix(),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {
        class_id: class_name
        for class_name, class_id in CLASS_TO_ID.items()
    },
}

yaml_path = OUTPUT_DIR / "data.yaml"

try:
    import yaml

    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data_yaml,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

except ImportError:
    # Minimal YAML writer if PyYAML is not installed.
    with yaml_path.open("w", encoding="utf-8") as file:
        file.write(f"path: '{data_yaml['path']}'\n")
        file.write("train: images/train\n")
        file.write("val: images/val\n")
        file.write("test: images/test\n")
        file.write("names:\n")
        for class_id, class_name in data_yaml["names"].items():
            file.write(f"  {class_id}: {class_name}\n")

print(yaml_path.read_text(encoding="utf-8"))

verification_records = []

for row in export_manifest.itertuples(index=False):
    image_path = Path(row.exported_image)
    label_path = Path(row.exported_label)

    result = {
        "image_id": row.image_id,
        "split": row.split,
        "image_exists": image_path.is_file(),
        "label_exists": label_path.is_file(),
        "valid_label_rows": 0,
        "invalid_label_rows": 0,
    }

    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()

            try:
                if len(parts) != 5:
                    raise ValueError("Expected five values")

                class_id = int(parts[0])
                coords = [float(value) for value in parts[1:]]

                if class_id not in CLASS_TO_ID.values():
                    raise ValueError("Unknown class ID")

                if not np.isfinite(coords).all():
                    raise ValueError("Non-finite coordinates")

                if not all(0 <= value <= 1 for value in coords):
                    raise ValueError("Coordinates outside [0, 1]")

                if coords[2] <= 0 or coords[3] <= 0:
                    raise ValueError("Non-positive box dimensions")

                result["valid_label_rows"] += 1

            except (ValueError, TypeError):
                result["invalid_label_rows"] += 1

    verification_records.append(result)

export_verification = pd.DataFrame(verification_records)

display(export_verification.groupby("split").agg(
    images=("image_exists", "sum"),
    labels=("label_exists", "sum"),
    valid_rows=("valid_label_rows", "sum"),
    invalid_rows=("invalid_label_rows", "sum"),
))

assert export_verification["image_exists"].all()
assert export_verification["label_exists"].all()
assert (export_verification["invalid_label_rows"] == 0).all()

print("All exported label files passed the format checks.")

split_manifest.to_csv(
    MANIFEST_DIR / "dataset_split_manifest.csv",
    index=False,
)

export_manifest.to_csv(
    MANIFEST_DIR / "yolo_export_manifest.csv",
    index=False,
)

export_verification.to_csv(
    REPORT_DIR / "export_verification.csv",
    index=False,
)

rejected_annotations.to_csv(
    REPORT_DIR / "rejected_annotations.csv",
    index=False,
)

excluded_images.to_csv(
    REPORT_DIR / "excluded_images_for_review.csv",
    index=False,
)

split_distribution.to_csv(
    REPORT_DIR / "split_class_distribution.csv",
)

print("Dataset preparation reports saved.")
print("Dataset YAML:", yaml_path)
print("Prepared dataset:", OUTPUT_DIR)