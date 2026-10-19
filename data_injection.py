
# data_injection.py
# Import statements
import hashlib
import json
import pandas as pd
from PIL import Image
from datetime import datetime, timezone
import math


from config import (
    PROJECT_DIR,
    DATASET_DIR,
    DATA_DIR,
    DATASET_NAME,
    DATASET_SUBSET,
    IMAGE_METADATA_FILE,
    OBJECT_ANNOTATIONS_FILE,
    IMAGE_EXTRACTION_ERRORS_FILE,
    ANNOTATION_EXTRACTION_ERRORS_FILE,
    validate_config,
)
#validate_config()
validate_config()


#  ===================================
# 1. Configuration and Output colums 
#  ==================================

Image_Metadata_Columns = [
    "image_id",
    "image_name",
    "image_path",
    "json_path",
    "image_width_px",
    "image_height_px",
    "image_format",
    "image_mode",
    "channel_count",
    "file_size_bytes",
    "sha256",
    "labelme_version",
    "json_image_width",
    "json_image_height",
    "annotation_count",
    "created_at",
]

ANNOTATION_COLUMNS = [
    "image_id",
    "annotation_id",
    "label",
    "shape_type",
    "points_json",
    "annotation_flags",
    "x1_px",
    "y1_px",
    "x2_px",
    "y2_px",
    "bbox_width_px",
    "bbox_height_px",
    "bbox_area_px2",
    "bbox_x_center_px",
    "bbox_y_center_px",
    "bbox_x_center_norm",
    "bbox_y_center_norm",
    "bbox_width_norm",
    "bbox_height_norm",
    "bbox_area_norm",
    "bbox_aspect_ratio",
    "out_of_bounds",
    "review_status",
    "created_at",
]

ERROR_COLUMNS = [
    "file_name",
    "file_path",
    "error_type",
    "error_message",
    "created_at",
]

# ===========================
# 2. Helper Functions
# ============================

def current_timestamp():
    """Returns the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()

def calculate_sha256(file_path):
    """Calculate the SHA-256 checksum without loading the whole file."""
    checksum = hashlib.sha256()

    with open(file_path,"rb") as file:
         for chunk in iter(lambda: file.read(1024 * 1024), b""):
            checksum.update(chunk) 

    return checksum.hexdigest()

def record_error(error_list, file_path, error_type, error_message):
    """Store an error without stopping the complete ingestion run."""
    error_list.append({
        "file_name": file_path.name,
        "file_path": str(file_path),
        "error_type": error_type,
        "error_message": str(error_message),
        "created_at": current_timestamp(),
    })

def is_finite_number(value):
    """Check that a coordinate is numeric and finite."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )

# ============================================================
# 3. EXTRACT IMAGE METADATA
# ============================================================
def generate_image_id(image_path):
    """Generate a namespaced image ID."""
    return calculate_sha256(image_path)[:16]

def extract_image_metadata(image_path, json_path, image_errors):
    """Read image metadata and its paired LabelMe JSON file."""

    try:
        if not json_path.is_file():
            raise FileNotFoundError(
                f"No matching JSON annotation file for {image_path.name}"
            )

        with Image.open(image_path) as image:
            image.load()

            image_width, image_height = image.size
            image_format = image.format
            image_mode = image.mode

            channel_count = len(image.getbands())

        with open(json_path, "r", encoding="utf-8-sig") as file:
            annotation_data = json.load(file)

        if not isinstance(annotation_data, dict):
            raise ValueError("JSON root must be an object.")

        shapes = annotation_data.get("shapes", [])

        if not isinstance(shapes, list):
            raise ValueError("'shapes' must be a list.")

        json_width = annotation_data.get("imageWidth")
        json_height = annotation_data.get("imageHeight")

        dimension_mismatch = (
            json_width != image_width
            or json_height != image_height
        )

        if dimension_mismatch:
            record_error(
                image_errors,
                json_path,
                "DIMENSION_MISMATCH",
                (
                    f"Image dimensions are {image_width}x{image_height}; "
                    f"JSON dimensions are {json_width}x{json_height}."
                ),
            )

        return {
            "image_id": generate_image_id(image_path),
            "image_name": image_path.name,
            "image_width_px": image_width,
            "image_height_px": image_height,
            "image_format": image_format,
            "image_mode": image_mode,
            "channel_count": channel_count,
            "file_size_bytes": image_path.stat().st_size,
            "labelme_version": annotation_data.get("version"),
            "json_image_width": json_width,
            "json_image_height": json_height,
            "annotation_count": len(shapes),
            "created_at": current_timestamp(),
            "sha256": calculate_sha256(image_path),
            "image_path": str(image_path.resolve()),
            "json_path": str(json_path.resolve()),
            
        }, annotation_data

    except Exception as error:
        record_error(
            image_errors,
            image_path,
            type(error).__name__,
            error,
        )
        return None, None

# ============================================================
# 4. EXTRACT OBJECT ANNOTATIONS
# ============================================================

def extract_annotations(
    image_path,
    annotation_data,
    image_width,
    image_height,
    annotation_errors,
):
    """Convert LabelMe shapes into structured annotation records."""

    image_id = image_path.stem
    shapes = annotation_data.get("shapes", [])
    image_records = []

    for index, shape in enumerate(shapes, start=1):
        try:
            if not isinstance(shape, dict):
                raise ValueError("Shape entry must be a JSON object.")

            label = shape.get("label")
            shape_type = shape.get("shape_type")
            points = shape.get("points", [])
            flags = shape.get("flags", {})

            if not isinstance(label, str) or not label.strip():
                raise ValueError("Annotation label is missing or empty.")

            if not isinstance(shape_type, str) or not shape_type:
                raise ValueError("shape_type is missing.")

            if not isinstance(points, list) or len(points) < 2:
                raise ValueError("At least two points are required.")

            if not isinstance(flags, dict):
                raise ValueError("'flags' must be a JSON object.")

            parsed_points = []

            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError(
                        f"Invalid point format: {point!r}"
                    )

                x, y = point

                if not is_finite_number(x) or not is_finite_number(y):
                    raise ValueError(
                        f"Coordinates must be finite numbers: {point!r}"
                    )

                parsed_points.append([float(x), float(y)])

            x_coordinates = [point[0] for point in parsed_points]
            y_coordinates = [point[1] for point in parsed_points]

            x1 = min(x_coordinates)
            y1 = min(y_coordinates)
            x2 = max(x_coordinates)
            y2 = max(y_coordinates)

            bbox_width = x2 - x1
            bbox_height = y2 - y1

            if bbox_width <= 0 or bbox_height <= 0:
                raise ValueError(
                    "Bounding-box width and height must be positive."
                )

            out_of_bounds = any(
                x < 0 or x > image_width
                or y < 0 or y > image_height
                for x, y in parsed_points
            )

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            bbox_area = bbox_width * bbox_height

            # A non-rectangular shape's bbox is only its enclosing rectangle.
            image_records.append({
                "annotation_id": f"{image_id}_{shape_type}_{index:06d}",
                "image_id": generate_image_id(image_path),
                "label": label.strip(),
                "shape_type": shape_type,
                "points_json": json.dumps(parsed_points),
                "annotation_flags": json.dumps(flags, sort_keys=True),
                "x1_px": x1,
                "y1_px": y1,
                "x2_px": x2,
                "y2_px": y2,
                "bbox_width_px": bbox_width,
                "bbox_height_px": bbox_height,
                "bbox_area_px2": bbox_area,
                "bbox_x_center_px": center_x,
                "bbox_y_center_px": center_y,
                "bbox_x_center_norm": center_x / image_width,
                "bbox_y_center_norm": center_y / image_height,
                "bbox_width_norm": bbox_width / image_width,
                "bbox_height_norm": bbox_height / image_height,
                "bbox_area_norm": bbox_area / (image_width * image_height),
                "bbox_aspect_ratio": bbox_width / bbox_height,
                "out_of_bounds": out_of_bounds,
                "review_status": (
                    "REVIEW"
                    if out_of_bounds
                    else "VALIDATION_PENDING"
                ),
                "created_at": current_timestamp(),
            })

        except Exception as error:
            record_error(
                annotation_errors,
                image_path.with_suffix(".json"),
                f"SHAPE_{index}_{type(error).__name__}",
                error,
            )

    return image_records

# ============================================================
# 5. MAIN DATA INJECTION PIPELINE
# ============================================================

def main():
    print("=" * 65)
    print("VISIONINSPECT AI - DATA INJECTION")
    print("=" * 65)

    # Validate configured paths and create the output directory.
    validate_config()

    image_records = []
    annotation_records = []
    image_errors = []
    annotation_errors = []

    # Discover PNG files directly inside the configured dataset folder.
    image_files = sorted(DATASET_DIR.glob("*.png"))

    if not image_files:
        raise FileNotFoundError(
            f"No PNG images found in: {DATASET_DIR}"
        )

    print(f"\nPNG images discovered: {len(image_files)}")

    # Track which JSON files are paired with discovered images.
    matched_json_paths = set()

    for image_path in image_files:
        json_path = image_path.with_suffix(".json")

        if json_path.is_file():
            matched_json_paths.add(json_path.resolve())

        metadata, annotation_data = extract_image_metadata(
            image_path,
            json_path,
            image_errors,
        )

        if metadata is None:
            continue

        image_records.append(metadata)

        extracted = extract_annotations(
            image_path,
            annotation_data,
            metadata["image_width_px"],
            metadata["image_height_px"],
            annotation_errors,
        )

        annotation_records.extend(extracted)

    # Find JSON files that do not have a corresponding PNG.
    for json_path in sorted(DATASET_DIR.glob("*.json")):
        if json_path.resolve() not in matched_json_paths:
            record_error(
                annotation_errors,
                json_path,
                "MISSING_IMAGE",
                "JSON annotation file has no matching PNG image.",
            )

    # Create DataFrames with fixed column names, even when empty.
    image_df = pd.DataFrame(
        image_records, 
        columns=Image_Metadata_Columns
    )
    annotation_df = pd.DataFrame(
        annotation_records,
        columns=ANNOTATION_COLUMNS,
    )
    image_error_df = pd.DataFrame(
        image_errors,
        columns=ERROR_COLUMNS,
    )
    annotation_error_df = pd.DataFrame(
        annotation_errors,
        columns=ERROR_COLUMNS,
    )

    # Write all four output CSV files.
    image_df.to_csv(IMAGE_METADATA_FILE, index=False, encoding="utf-8")
    annotation_df.to_csv(
        OBJECT_ANNOTATIONS_FILE,
        index=False,
        encoding="utf-8",
    )
    image_error_df.to_csv(
        IMAGE_EXTRACTION_ERRORS_FILE,
        index=False,
        encoding="utf-8",
    )
    annotation_error_df.to_csv(
        ANNOTATION_EXTRACTION_ERRORS_FILE,
        index=False,
        encoding="utf-8",
    )

    # Report ingestion results.
    print("\n" + "=" * 65)
    print("DATA INJECTION SUMMARY")
    print("=" * 65)
    print(f"PNG images discovered:       {len(image_files)}")
    print(f"Image metadata rows:         {len(image_df)}")
    print(f"Object annotation rows:      {len(annotation_df)}")
    print(f"Image extraction errors:     {len(image_error_df)}")
    print(f"Annotation extraction errors:{len(annotation_error_df)}")

    if not annotation_df.empty:
        print("\nAnnotation counts by label:")
        print(annotation_df["label"].value_counts().to_string())

        print(
            "\nAnnotations outside image boundaries: "
            f"{int(annotation_df['out_of_bounds'].sum())}"
        )

        print(
            "Duplicate annotation IDs: "
            f"{int(annotation_df['annotation_id'].duplicated().sum())}"
        )

    print("\nOutput files:")
    print(f"  Images:      {IMAGE_METADATA_FILE}")
    print(f"  Annotations: {OBJECT_ANNOTATIONS_FILE}")
    print(f"  Image errors:{IMAGE_EXTRACTION_ERRORS_FILE}")
    print(f"  JSON errors: {ANNOTATION_EXTRACTION_ERRORS_FILE}")

    print("\nData injection finished.")


if __name__ == "__main__":
    main()
