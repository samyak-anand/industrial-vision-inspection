
# Industrial Vision Inspection

**An end-to-end computer vision project for industrial surface-defect inspection and anomaly detection.**

## Overview

Industrial quality inspection requires reliable identification of defects, consistent evaluation, and traceable inspection results. This project aims to develop a computer vision platform that detects anomalous product surfaces and presents the results through a usable inspection interface.

The project uses the MVTec Anomaly Detection (MVTec AD) dataset to investigate anomaly detection on industrial images, starting with the `metal_nut` category.

## Objectives

- Explore and validate an industrial image dataset.
- Establish a reproducible anomaly-detection baseline.
- Compare candidate models using appropriate image-level and pixel-level metrics.
- Visualize anomaly maps and analyze false positives and missed defects.
- Expose inference through a REST API.
- Build an inspection dashboard with persistent prediction history.
- Add automated tests and reproducible local deployment.

## Planned Technology Stack

- **Computer Vision:** PyTorch, OpenCV
- **Anomaly Detection:** PaDiM / PatchCore evaluation
- **API:** FastAPI, Pydantic
- **Dashboard:** Streamlit
- **Database:** PostgreSQL
- **Testing:** pytest
- **Deployment:** Docker Compose
- **Experiment Tracking:** MLflow, if implemented

## Dataset

Primary dataset: [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad)

The initial experiment focuses on industrial anomaly detection using normal training images and labeled test images with ground-truth defect masks.

## Project Status

**Status:** In development

The planned capabilities above will be marked as implemented only after they have been built and tested. Model metrics, screenshots, deployment instructions, and limitations will be documented as the project progresses.

## Scope and Limitations

This is a portfolio research and engineering prototype using a public benchmark dataset. Results on benchmark images do not establish readiness for real factory deployment. Industrial validation would require testing with representative production images, cameras, lighting, product variants, and operating conditions.
