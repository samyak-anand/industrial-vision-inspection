#config.py

from pathlib import Path

# Define the base directory for the project 

PROJECT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------
# External dataset (NOT inside GitHub repository)
# --------------------------------------------------
DATASET_DIR = (
    PROJECT_DIR.parent
    / "dataset-ind-vision-insp"
    / "screw_candidate_detection"
    / "candidate detection"
)
# --------------------------------------------------
# Generated outputs inside the repository
# --------------------------------------------------
DATA_DIR = PROJECT_DIR / "data"

IMAGE_METADATA_FILE = DATA_DIR / "image_metadata.csv"
OBJECT_ANNOTATIONS_FILE = DATA_DIR / "object_annotations.csv"

IMAGE_EXTRACTION_ERRORS_FILE = (
    DATA_DIR / "image_extraction_errors.csv"
)

ANNOTATION_EXTRACTION_ERRORS_FILE = (
    DATA_DIR / "annotation_extraction_errors.csv"
)

IMAGE_EXTENSION = ".png"
ANNOTATION_EXTENSION = ".json"


def validate_config():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Project: {PROJECT_DIR}")
    print(f"External dataset: {DATASET_DIR}")
    print(f"Output directory: {DATA_DIR}")

print("Project root:", PROJECT_DIR)
print("Dataset directory:", DATASET_DIR)
print("Dataset exists:", DATASET_DIR.exists())
print("Output directory:", DATA_DIR)
print("Output directory exists:", DATA_DIR.exists())

if __name__ == "__main__":
    validate_config()