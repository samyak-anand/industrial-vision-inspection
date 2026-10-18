
# data_injection.py

from config import (
    PROJECT_DIR,
    DATASET_DIR,
    DATA_DIR,
    validate_config,
)


def main():
    print("\n" + "=" * 60)
    print("VisionInspect AI - Data Injection")
    print("=" * 60)

    # Validate the configured dataset and output directory.
    validate_config()

    # Confirm expected input and output paths.
    print(f"\nProject root: {PROJECT_DIR}")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Output directory: {DATA_DIR}")

    # Confirm the expected source directory exists.
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    print("\nPath validation completed successfully.")
    print("Dataset is accessible.")
    print("Output directory is ready.")
    print("No dataset files have been modified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
