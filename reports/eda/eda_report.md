# VisionInspect AI — EDA Summary

- Analysis time (UTC): 2026-10-20T07:22:11.724052+00:00
- Project root: `C:\Users\samya\PyCharmProject\industrial-vision-inspection`
- Dataset directory (configured): `C:\Users\samya\PyCharmProject\dataset-ind-vision-insp\screw_candidate_detection\candidate detection`

## Input files
- **images:** `C:\Users\samya\PyCharmProject\industrial-vision-inspection\data\image_metadata.csv` — exists
- **annotations:** `C:\Users\samya\PyCharmProject\industrial-vision-inspection\data\object_annotations.csv` — exists
- **image_errors:** `C:\Users\samya\PyCharmProject\industrial-vision-inspection\data\image_extraction_errors.csv` — exists
- **annotation_errors:** `C:\Users\samya\PyCharmProject\industrial-vision-inspection\data\annotation_extraction_errors.csv` — exists

## Dataset sizes

- **images:** 821 rows × 17 columns
- **annotations:** 10,226 rows × 24 columns
- **image_errors:** 0 rows × 5 columns
- **annotation_errors:** 0 rows × 5 columns

## Dataset summary

| metric                                     |   value |
|:-------------------------------------------|--------:|
| image_rows                                 |     821 |
| unique_nonmissing_image_ids                |     821 |
| image_rows_with_duplicate_id               |       0 |
| annotation_rows                            |   10226 |
| unique_nonmissing_annotation_ids           |   10226 |
| annotation_rows_with_duplicate_id          |       0 |
| unique_images_with_annotations             |     821 |
| image_ids_without_matching_annotation      |       0 |
| annotation_image_ids_not_in_image_metadata |       0 |
| image_extraction_error_rows                |       0 |
| annotation_extraction_error_rows           |       0 |

## Label distribution

| label   |   annotation_count |   annotation_pct |   images_with_label |
|:--------|-------------------:|-----------------:|--------------------:|
| screw   |               7941 |           77.655 |                 796 |
| hole    |               2285 |           22.345 |                 489 |

## Readiness checklist

| check                          | status       | evidence                                                                   | rule_or_limitation                                                                                           |
|:-------------------------------|:-------------|:---------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------|
| Required CSV files             | PASS         | 4/4 files exist                                                            | PASS only when all four configured CSV inputs exist.                                                         |
| Required columns               | FAIL         | 6 missing required column(s)                                               | PASS when required image and annotation fields are present.                                                  |
| Extraction errors              | PASS         | 0 image errors; 0 annotation errors                                        | WARNING if any extraction errors exist; inspect error CSVs before deciding whether records can be recovered. |
| Identifier uniqueness          | PASS         | 0 rows belong to duplicate ID groups                                       | PASS when image_id and annotation_id are unique within their respective tables.                              |
| Annotation-to-image references | PASS         | 0 annotation image ID(s) absent from image metadata                        | PASS when every annotation image_id matches an image metadata record.                                        |
| Train/validation/test split    | NOT ASSESSED | No populated split assignment found                                        | A split must be deliberately created, preferably grouped by source/image identity to avoid leakage.          |
| Visual annotation verification | NOT ASSESSED | Overlay review must be completed by inspecting saved sample visualizations | Human review is required to confirm box alignment, label correctness, and missed objects.                    |

## Limitations and next steps

- Outliers are candidates for manual review, not automatic errors.
- Correlations among derived geometry fields are descriptive and do not measure detector accuracy.
- The extraction pipeline converts each LabelMe shape to its point-extrema bounding rectangle; this is not the same as polygon area.
- Confirm the annotation policy for images with zero annotations.
- Review extraction errors and rendered annotation overlays.
- Establish leakage-resistant train/validation/test splits before training.
- Evaluate the first detector on held-out images with agreed precision, recall, false-accept and false-reject criteria.