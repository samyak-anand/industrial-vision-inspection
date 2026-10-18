
# data_injection.py
# Import statements
import hashlib
import json
import pandas as pd
from PIL import Image
from datetime import datetime, timezone

from config import (
    PROJECT_DIR,
    DATASET_DIR,
    DATA_DIR,
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


