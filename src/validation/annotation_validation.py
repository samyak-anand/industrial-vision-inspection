#annotation_validation.py
#import lib
from pathlib import Path
import json
import sys
from collections import Counter
import pandas as pd
import numpy as np
from PIL import Image
from IPython.display import display

#Root path 
PROJECT_ROOT =Path(
     r"C:\Users\samya\PyCharmProject\industrial-vision-inspection"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from config import (
    DATASET_DIR,
    DATA_DIR,
    IMAGE_METADATA_FILE,
    OBJECT_ANNOTATIONS_FILE,
)
print("Project root:", PROJECT_ROOT)
print("Dataset directory:", DATASET_DIR)
print("Output directory:", DATA_DIR)

assert PROJECT_ROOT.is_dir(), "Project directory does not exist."
assert DATASET_DIR.is_dir(), "Dataset directory does not exist."
assert DATA_DIR.is_dir(), "Output directory does not exist."

metadata_path = Path(IMAGE_METADATA_FILE)
annotations_path = Path(OBJECT_ANNOTATIONS_FILE)

for path in [metadata_path, annotations_path]:
    print(f"{path.name}: {'FOUND' if path.is_file() else 'MISSING'}")

image_files = sorted(
    p for p in DATASET_DIR.iterdir()
    if p.is_file() and p.suffix.lower() == ".png"
)

json_files = sorted(
    p for p in DATASET_DIR.iterdir()
    if p.is_file() and p.suffix.lower() == ".json"
)

image_stems = {p.stem for p in image_files}
json_stems = {p.stem for p in json_files}

inventory = pd.Series({
    "PNG images": len(image_files),
    "JSON annotations": len(json_files),
    "Images without JSON": len(image_stems - json_stems),
    "JSON without image": len(json_stems - image_stems),
    "Matching image-JSON pairs": len(image_stems & json_stems),
}, name="count")

display(inventory.to_frame())

import math

SUPPORTED_LABELS = {"screw", "hole"}

validation_records = []
shape_records = []

for json_path in json_files:
    image_path = DATASET_DIR / f"{json_path.stem}.png"

    record = {
        "image_id": json_path.stem,
        "image_path": str(image_path),
        "json_path": str(json_path),
        "image_exists": image_path.is_file(),
        "json_valid": False,
        "image_readable": False,
        "dimension_match": False,
        "shape_count": 0,
        "rectangle_count": 0,
        "unsupported_shape_count": 0,
        "unknown_label_count": 0,
        "invalid_coordinate_count": 0,
        "invalid_box_count": 0,
        "out_of_bounds_count": 0,
        "error": None,
    }

    try:
        with json_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        record["json_valid"] = isinstance(data, dict)

        if not record["json_valid"]:
            raise ValueError("JSON root must be an object.")

        width = data.get("imageWidth")
        height = data.get("imageHeight")
        shapes = data.get("shapes", [])

        if (
            not isinstance(width, int) or width <= 0
            or not isinstance(height, int) or height <= 0
        ):
            raise ValueError("Invalid imageWidth or imageHeight.")

        if not isinstance(shapes, list):
            raise ValueError("'shapes' must be a list.")

        record["shape_count"] = len(shapes)

        if image_path.is_file():
            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    actual_width, actual_height = image.size

                record["image_readable"] = True
                record["dimension_match"] = (
                    actual_width == width and actual_height == height
                )

            except Exception as exc:
                record["error"] = f"Image error: {exc}"

        for index, shape in enumerate(shapes):
            label = str(shape.get("label", "")).strip().lower()
            shape_type = shape.get("shape_type", "")
            points = shape.get("points", [])

            item = {
                "image_id": json_path.stem,
                "annotation_index": index,
                "annotation_id": f"{json_path.stem}_{index + 1:04d}",
                "label": label,
                "shape_type": shape_type,
                "image_width": width,
                "image_height": height,
                "valid_coordinates": False,
                "valid_box": False,
                "out_of_bounds": False,
                "x_min": None,
                "y_min": None,
                "x_max": None,
                "y_max": None,
            }

            if label not in SUPPORTED_LABELS:
                record["unknown_label_count"] += 1

            if shape_type != "rectangle":
                record["unsupported_shape_count"] += 1
                shape_records.append(item)
                continue

            record["rectangle_count"] += 1

            try:
                if not isinstance(points, list) or len(points) != 2:
                    raise ValueError("Rectangle must have two corner points.")

                coords = [
                    float(value)
                    for point in points
                    for value in point
                ]

                if (
                    len(coords) != 4
                    or not all(math.isfinite(v) for v in coords)
                ):
                    raise ValueError("Coordinates must be finite numbers.")

                x1, y1, x2, y2 = coords

                item.update({
                    "x_min": min(x1, x2),
                    "y_min": min(y1, y2),
                    "x_max": max(x1, x2),
                    "y_max": max(y1, y2),
                    "valid_coordinates": True,
                })

                record["invalid_coordinate_count"] += 0

                x_min, y_min = item["x_min"], item["y_min"]
                x_max, y_max = item["x_max"], item["y_max"]

                if x_max <= x_min or y_max <= y_min:
                    record["invalid_box_count"] += 1
                else:
                    item["valid_box"] = True

                if (
                    x_min < 0 or y_min < 0
                    or x_max > width or y_max > height
                ):
                    item["out_of_bounds"] = True
                    record["out_of_bounds_count"] += 1

            except (TypeError, ValueError, IndexError) as exc:
                record["invalid_coordinate_count"] += 1
                item["coordinate_error"] = str(exc)

            shape_records.append(item)

    except Exception as exc:
        record["error"] = str(exc)

    validation_records.append(record)

image_validation = pd.DataFrame(validation_records)
annotation_validation = pd.DataFrame(shape_records)

print("Images processed:", len(image_validation))
print("Annotation records processed:", len(annotation_validation))

display(image_validation.head())
display(annotation_validation.head())

image_quality_summary = pd.Series({
    "Total JSON files": len(image_validation),
    "Invalid JSON or structure": (
        ~image_validation["json_valid"]
    ).sum(),
    "Missing images": (
        ~image_validation["image_exists"]
    ).sum(),
    "Unreadable images": (
        image_validation["image_exists"]
        & ~image_validation["image_readable"]
    ).sum(),
    "Image dimension mismatches": (
        image_validation["image_readable"]
        & ~image_validation["dimension_match"]
    ).sum(),
    "Images with unsupported shapes": (
        image_validation["unsupported_shape_count"] > 0
    ).sum(),
    "Images with unknown labels": (
        image_validation["unknown_label_count"] > 0
    ).sum(),
    "Images with invalid boxes": (
        image_validation["invalid_box_count"] > 0
    ).sum(),
    "Images with out-of-bounds boxes": (
        image_validation["out_of_bounds_count"] > 0
    ).sum(),
}, name="value")

display(image_quality_summary.to_frame())

class_distribution = (
    annotation_validation["label"]
    .value_counts(dropna=False)
    .rename_axis("label")
    .reset_index(name="annotation_count")
)

display(class_distribution)

print("Shape types:")
display(
    annotation_validation["shape_type"]
    .value_counts(dropna=False)
    .rename("count")
    .to_frame()
)

print("Invalid annotations:")
invalid_annotations = annotation_validation[
    (~annotation_validation["valid_coordinates"])
    | (
        (annotation_validation["shape_type"] == "rectangle")
        & (
            ~annotation_validation["valid_box"]
            | annotation_validation["out_of_bounds"]
        )
    )
]

display(invalid_annotations.head(20))
metadata_df = pd.read_csv(metadata_path)
annotations_df = pd.read_csv(annotations_path)

print("Image metadata rows:", len(metadata_df))
print("Extracted annotation rows:", len(annotations_df))
print("Raw JSON shape records:", len(annotation_validation))

print("\nMetadata columns:")
print(metadata_df.columns.tolist())

print("\nAnnotation columns:")
print(annotations_df.columns.tolist())

display(metadata_df.head())
display(annotations_df.head())

REPORT_DIR = DATA_DIR / "validation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

image_validation.to_csv(
    REPORT_DIR / "image_validation_report.csv",
    index=False
)

annotation_validation.to_csv(
    REPORT_DIR / "annotation_validation_report.csv",
    index=False
)

image_quality_summary.to_csv(
    REPORT_DIR / "image_quality_summary.csv",
    header=True
)

class_distribution.to_csv(
    REPORT_DIR / "class_distribution.csv",
    index=False
)

invalid_annotations.to_csv(
    REPORT_DIR / "invalid_annotations.csv",
    index=False
)

print("Validation reports saved to:", REPORT_DIR)
blocking_issues = {
    "invalid_json": int((~image_validation["json_valid"]).sum()),
    "missing_images": int((~image_validation["image_exists"]).sum()),
    "unreadable_images": int((
        image_validation["image_exists"]
        & ~image_validation["image_readable"]
    ).sum()),
    "dimension_mismatches": int((
        image_validation["image_readable"]
        & ~image_validation["dimension_match"]
    ).sum()),
    "invalid_coordinates": int(
        image_validation["invalid_coordinate_count"].sum()
    ),
    "invalid_boxes": int(
        image_validation["invalid_box_count"].sum()
    ),
    "out_of_bounds_boxes": int(
        image_validation["out_of_bounds_count"].sum()
    ),
    "unknown_labels": int(
        image_validation["unknown_label_count"].sum()
    ),
    "unsupported_shapes": int(
        image_validation["unsupported_shape_count"].sum()
    ),
}

for issue, count in blocking_issues.items():
    print(f"{issue}: {count}")

print("\nValidation complete. Review flagged records before dataset preparation.")

print("IMAGE METADATA COLUMNS")
print(metadata_df.columns.tolist())

print("\nANNOTATION COLUMNS")
print(annotations_df.columns.tolist())

print("\nANNOTATION DATA TYPES")
display(annotations_df.dtypes.astype(str).to_frame("dtype"))

print("\nMISSING VALUES")
display(
    annotations_df.isna()
    .sum()
    .sort_values(ascending=False)
    .to_frame("missing_count")
)