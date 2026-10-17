import pypandoc
from pathlib import Path

readme = r'''# VisionInspect AI
### Industrial Machine-Vision Inspection for Surface Defects, Colour Conformity & Screw Verification

<p align="center">
  <strong>A modular, traceable and evaluation-driven blueprint for AI-assisted industrial quality inspection.</strong>
</p>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/Project-Prototype%20in%20development-orange">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-blue">
  <img alt="Database" src="https://img.shields.io/badge/Database-PostgreSQL-4169E1">
  <img alt="Vision" src="https://img.shields.io/badge/Computer%20Vision-OpenCV%20%7C%20PyTorch-5C3EE8">
  <img alt="License" src="https://img.shields.io/badge/License-To%20be%20defined-lightgrey">
</p>

> **Project status:** Design and proof-of-concept development. This repository describes the proposed architecture and implementation roadmap. A component should be described as *implemented* only after it has been built and tested. No production accuracy, inspection-cycle-time, or industrial-readiness claim is made by this document.

---

## Table of contents

- [1. Project definition](#1-project-definition)
- [2. Problem statement](#2-problem-statement)
- [3. Aim and objectives](#3-aim-and-objectives)
- [4. Scope and use cases](#4-scope-and-use-cases)
- [5. Proposed system at a glance](#5-proposed-system-at-a-glance)
- [6. Proposed architecture](#6-proposed-architecture)
- [7. Inspection workflow and data flow](#7-inspection-workflow-and-data-flow)
- [8. Machine-vision strategy](#8-machine-vision-strategy)
- [9. Database design](#9-database-design)
- [10. Proposed database schema](#10-proposed-database-schema)
- [11. Repository structure](#11-repository-structure)
- [12. Components and responsibilities](#12-components-and-responsibilities)
- [13. Hardware architecture](#13-hardware-architecture)
- [14. Software stack and requirements](#14-software-stack-and-requirements)
- [15. Python libraries](#15-python-libraries)
- [16. Dataset strategy](#16-dataset-strategy)
- [17. Evaluation and experimentation](#17-evaluation-and-experimentation)
- [18. Synthetic data and baseline plan](#18-synthetic-data-and-baseline-plan)
- [19. Installation and local setup](#19-installation-and-local-setup)
- [20. Configuration and environment variables](#20-configuration-and-environment-variables)
- [21. Testing and quality assurance](#21-testing-and-quality-assurance)
- [22. Security, reliability and traceability](#22-security-reliability-and-traceability)
- [23. Limitations and assumptions](#23-limitations-and-assumptions)
- [24. Development roadmap](#24-development-roadmap)
- [25. Definition of done](#25-definition-of-done)
- [26. References](#26-references)

---

## 1. Project definition

**VisionInspect AI** is a proposed industrial machine-vision inspection system for evaluating appliance cabinets and similar manufactured components. It combines image acquisition, image-quality checks, computer-vision inference, inspection rules, structured database records, and operator-facing results into a traceable inspection workflow.

The intended system is designed around three inspection needs:

1. **Surface-defect inspection** — identify or localise visible defects such as scratches, dents, stains, coating irregularities, and other surface anomalies where the camera setup and image resolution make them observable.
2. **Colour conformity** — compare the observed product colour with an approved reference or tolerance specification under controlled illumination and calibrated imaging conditions.
3. **Screw and component verification** — verify the presence, approximate location, and where required the orientation or type of specified screws and visible components.

These are related but technically different tasks. Surface anomalies may be approached using anomaly detection, segmentation, or object detection. Colour conformity requires controlled colour measurement and tolerance rules. Screw verification is generally a detection or localisation problem. The final method for each task must be selected and validated against representative production images.

### Intended users

- Quality-assurance and quality-control engineers
- Machine-vision and automation engineers
- Data scientists and computer-vision engineers
- Production and manufacturing engineers
- Application developers responsible for inspection interfaces and traceability

### Design principle

The system should not store only a final `PASS` or `FAIL`. It should preserve the evidence needed to understand that decision: product and inspection identifiers, image references, model and configuration versions, defect detections, measured values, decision rules, timestamps, and review history.

---

## 2. Problem statement

Manual visual inspection can be affected by operator fatigue, subjective judgement, inconsistent lighting, product variation, and differences in defect interpretation. A conventional rule-based vision system can be effective for tightly controlled conditions, but may become difficult to maintain when appearance varies or when defects are subtle.

A machine-learning model alone does not solve the entire manufacturing problem. A usable inspection solution also needs repeatable image capture, reliable inference, clear decision rules, traceable storage, review of uncertain results, and integration with the production process.

VisionInspect AI therefore addresses the following problem:

> **How can a repeatable, measurable and traceable inspection workflow be designed to identify visible product defects and conformity issues, record the evidence behind each decision, and provide a foundation for later integration with industrial equipment?**

### Main engineering challenges

- Collecting representative images of normal products and relevant defect types.
- Separating real defects from reflections, shadows, texture, dust, colour variation, and imaging artefacts.
- Maintaining stable image capture across lighting, camera, product, and production changes.
- Handling rare defects and class imbalance without relying on misleading aggregate accuracy.
- Choosing task-appropriate metrics and thresholds that balance false acceptance and false rejection.
- Linking every inspection result to the correct product, image, model version, and production event.
- Meeting the required cycle time without sacrificing inspection reliability.
- Designing a prototype that can be tested locally while keeping the future industrial deployment path clear.

---

## 3. Aim and objectives

### Aim

Develop a reproducible proof of concept for an AI-assisted industrial inspection pipeline that demonstrates image processing, model evaluation, structured inspection records, and an end-to-end data flow.

### Technical objectives

1. Establish a documented and repeatable baseline for visual anomaly detection.
2. Start with the `metal_nut` category of the public MVTec Anomaly Detection (MVTec AD) benchmark as an initial software experiment.
3. Validate input images, labels, masks, metadata, and train/test separation before training or evaluation.
4. Compare appropriate baseline methods before attempting more complex models.
5. Evaluate image-level anomaly detection and, where applicable, pixel-level localisation.
6. Examine false positives, missed defects, anomaly maps, and threshold sensitivity.
7. Design and implement a PostgreSQL data model for products, inspections, image references, model versions, predictions, and review/audit events.
8. Integrate the inspection pipeline through an API and a simple operator or engineering dashboard.
9. Add automated tests, configuration management, logging, and reproducible experiment records.
10. Document limitations and define the evidence required before a factory pilot.

### Success criteria for the proof of concept

The proof of concept should be considered successful when it can, at minimum:

- Run the same documented experiment from a clean environment.
- Produce evaluation results on a held-out test set without data leakage.
- Show representative correct detections, false alarms, and missed defects.
- Persist an inspection record and its model/configuration references in PostgreSQL.
- Retrieve and display prior inspection results in a basic interface.
- Run automated tests for critical data validation and API/database behaviour.
- Clearly distinguish measured results from targets and future work.

Numerical acceptance thresholds should be agreed with the project owner and validated against the cost of false acceptance, false rejection, and the production cycle-time requirement. They are not assumed in advance.

---

## 4. Scope and use cases

### Initial proof-of-concept scope

| Area | Initial scope | Later extension |
|---|---|---|
| Surface anomalies | MVTec AD `metal_nut` benchmark and baseline evaluation | Representative appliance-cabinet images and additional defect types |
| Colour conformity | Design for reference colour, measured colour, and tolerance records | Calibrated colour workflow validated under production lighting |
| Screw verification | Define expected component/position records and integration boundary | Train and validate a dedicated detector on labelled production images |
| Database | PostgreSQL schema and inspection traceability | Production retention, backup, access control, and high availability |
| API | Local inference and inspection-result endpoints | Authentication, monitoring, deployment hardening |
| Dashboard | Engineering-oriented result/history view | Operator workflow, review queue, role-based access |
| Equipment | Software prototype using stored images | Industrial camera, lighting, trigger, PLC and line integration |

### Example user stories

- A quality engineer can open an inspection record and see its decision, image reference, detected anomalies, model version, and timestamp.
- A developer can compare experiment configurations and evaluate how a threshold changes false-positive and false-negative behaviour.
- An operator can review an uncertain result and record a disposition with a reason.
- An engineer can identify which model version generated a particular decision.
- A system maintainer can trace an API or database failure using structured logs and a correlation identifier.

---

## 5. Proposed system at a glance
<img width="1425" height="1401" alt="image" src="https://github.com/user-attachments/assets/af130728-4ad5-4189-97e1-839e9163cb8f" />

```text
Product / Test Image
        |
        v
Image Intake and Validation
        |
        v
Pre-processing and Image-Quality Checks
        |
        v
Task Router
  |          |             |
  v          v             v
Surface     Colour        Screw / Component
Anomaly     Conformity    Verification
Model       Measurement   Detector
  |          |             |
  +----------+-------------+
             |
             v
      Decision / Rules Layer
             |
       +-----+------+
       |            |
       v            v
   PASS / FAIL   REVIEW REQUIRED
       |            |
       +-----+------+
             |
             v
 Inspection API and PostgreSQL
             |
             v
 Dashboard / History / Audit Trail
```

This is a logical design, not a claim that all blocks currently exist in the repository. A first implementation should keep the software boundaries small and add integration incrementally.

---

## 6. Proposed architecture

The architecture is divided into layers so that the vision model can be developed independently of the database, user interface, and future factory equipment.

### 6.1 Image acquisition layer

**Purpose:** obtain a stable image and associate it with a product or inspection event.

Responsibilities:

- Accept a file or image reference in the initial prototype.
- In a future cell, acquire an image from a triggered industrial camera.
- Capture acquisition timestamp, camera identifier, product identifier, and capture configuration where available.
- Reject unreadable, empty, corrupt, or incorrectly sized images.
- Preserve the original image or a governed reference to it.

### 6.2 Image validation and pre-processing layer

**Purpose:** make input quality and transformation steps explicit.

Responsibilities:

- Validate image format, dimensions, channel count, and file integrity.
- Apply documented resizing, normalisation, cropping, or region-of-interest extraction.
- Check for conditions such as blur, overexposure, underexposure, and incomplete product framing where suitable quality checks are implemented.
- Record preprocessing configuration and version.
- Avoid transformations that remove defect evidence or invalidate colour measurement.

### 6.3 Computer-vision inference layer

**Purpose:** produce task-specific outputs from validated images.

Potential approaches:

- **Surface anomaly detection:** PaDiM or PatchCore as candidate baselines; other methods can be assessed later.
- **Defect localisation:** segmentation or anomaly-map methods where pixel-level labels or reliable ground truth are available.
- **Screw verification:** an object detector or a controlled geometric/feature-based method, selected using representative labelled images.
- **Colour conformity:** calibrated colour-space measurements and tolerance comparison, rather than assuming an RGB threshold is universally reliable.

Each inference response should include a model identifier/version, processing status, scores or measurements, and the relevant output references.

### 6.4 Decision and business-rules layer

**Purpose:** convert model outputs into a controlled inspection disposition.

Possible dispositions:

- `PASS` — all required checks satisfy approved rules.
- `FAIL` — at least one confirmed rule violation exists.
- `REVIEW` — confidence, image quality, or evidence is insufficient for an automatic decision.
- `ERROR` — inspection could not be completed due to a technical failure.

Thresholds should be configuration-controlled and evaluated using validation data. `REVIEW` is important: the system should not force uncertain predictions into a pass/fail decision without an agreed risk policy.

### 6.5 API and application layer

**Purpose:** provide a stable interface for submitting inspections and retrieving results.

Proposed API responsibilities:

- Accept an inspection request or image reference.
- Validate request payloads.
- Call preprocessing and inference services.
- Apply decision rules.
- Persist inspection outputs and status.
- Return a structured response with an inspection identifier.
- Expose read endpoints for history and result details.

FastAPI and Pydantic are proposed options. Endpoint names and request schemas should be finalised during implementation.

### 6.6 Persistence and traceability layer

**Purpose:** retain structured records and relationships.

PostgreSQL should store metadata, inspection events, prediction summaries, model versions, rules, and audit history. Large image files and pixel-level output artefacts may be stored in a filesystem or object store, with a URI and checksum in the database. Storing every image as a large database binary is not the default recommendation.

### 6.7 Dashboard and review layer

**Purpose:** allow engineering users to inspect outcomes and investigate errors.

Potential views:

- Recent inspection history and status counts
- Image with anomaly overlay or defect bounding boxes
- Product and inspection metadata
- Model version and inference time
- Defect type, score, location, and decision reason
- Manual review/disposition history
- Model evaluation summaries and experiment comparisons

Streamlit is a candidate for a lightweight prototype dashboard. A production operator interface may require a different technology and industrial usability review.

### 6.8 Observability and experiment layer

**Purpose:** make behaviour measurable and debugging reproducible.

Capture, where available:

- Request/correlation ID
- Input validation failures
- Model and preprocessing versions
- Inference latency and total request latency
- Database errors and retry outcomes
- Model evaluation configuration and results
- Software and dependency versions

MLflow may be introduced for experiment tracking if it adds value to the implemented workflow.

---

## 7. Inspection workflow and data flow

1. **Create an inspection request.** Assign or receive an inspection ID and associate it with a product, batch, or test case.
2. **Register the image.** Store the original image or a controlled file reference; calculate a checksum where practical.
3. **Validate the input.** Check readability, dimensions, expected view, and required metadata.
4. **Pre-process the image.** Apply the versioned transformation configuration.
5. **Run the relevant inspection task.** Invoke the selected model or measurement procedure.
6. **Apply decision rules.** Compare the output with versioned thresholds and required checks.
7. **Persist the result.** Store status, scores, detections, timings, model version, and relevant references.
8. **Expose the result.** Return the inspection ID and result through the API; display it in the dashboard.
9. **Review when required.** Capture the reviewer, decision, timestamp, and reason.
10. **Monitor and improve.** Analyse false positives, missed defects, drift, latency, and changes in production conditions.

A retry should not silently create duplicate inspection outcomes. Request idempotency and transaction behaviour should be defined when implementing the API and persistence layer.

---

## 8. Machine-vision strategy

Different inspection requirements need different measurement and modelling strategies.

### 8.1 Surface-defect inspection

Initial research direction: anomaly detection on MVTec AD, beginning with `metal_nut`.

- Train using normal images where the selected method supports unsupervised or one-class training.
- Evaluate on the dataset's test split, which includes normal and anomalous examples.
- Use ground-truth masks for pixel-level evaluation where supplied.
- Review anomaly maps to determine whether highlighted regions correspond to defects or nuisance variation.
- Record false-positive examples and missed defects rather than relying on a single aggregate metric.

**Candidate baseline models:** PaDiM and PatchCore. These are candidates for evaluation, not a statement that either has already been trained or selected.

### 8.2 Colour conformity

Colour inspection is sensitive to camera response, white balance, illumination spectrum, exposure, surface reflectance, and viewing geometry.

Proposed workflow:

1. Control and document illumination and camera settings.
2. Use a suitable colour reference and calibration procedure.
3. Define the reference colour and allowed tolerance with the quality team.
4. Convert measurements into an agreed colour space, potentially CIE Lab.
5. Select a difference metric, such as Delta E, appropriate to the specification.
6. Validate repeatability across products, cameras, and operating conditions.

Do not claim colour conformity from uncalibrated RGB values alone.

### 8.3 Screw and component verification

Potential checks include presence/absence, location, orientation, type, and count.

The implementation requires labelled images that represent the actual screw types, viewpoints, product variants, occlusion, and acceptable assembly variation. Detection confidence alone should not be treated as a complete acceptance policy. The decision rules must define which positions are mandatory and how uncertain detections are handled.

### 8.4 Multi-task inspection

A later system can combine outputs from separate task-specific modules. The overall product decision should be explainable: for example, a failed screw check or an out-of-tolerance colour measurement should be traceable to the corresponding rule and evidence.

---

## 9. Database design

### 9.1 Database objectives

The proposed PostgreSQL data model is intended to support:

- Product, product variant, batch, and inspection traceability
- Image metadata and file/object references
- Inspection requests, lifecycle state, and final disposition
- Model registry metadata and version tracking
- Inference runs, anomaly scores, detections, and measurements
- Decision rules and threshold/configuration versions
- Manual review and audit events
- Evaluation runs and aggregate metrics
- Referential integrity and repeatable queries

### 9.2 Design principles

- **Normalised core records:** keep product, model, inspection, and result entities separate to reduce duplication.
- **Stable identifiers:** use primary keys for entities and foreign keys for relationships.
- **UTC timestamps:** store event times consistently, preferably with PostgreSQL `timestamptz`.
- **Explicit units:** record units for measurements and processing times; avoid ambiguous numeric fields.
- **Versioned configuration:** retain model version, preprocessing version, threshold/rule version, and experiment configuration.
- **Traceability:** every prediction should link to the inspection and model version that produced it.
- **Controlled status values:** use constraints, lookup tables, or PostgreSQL enums where justified.
- **Image separation:** store large image/artefact bytes outside the relational database by default; retain a URI, checksum, format, and metadata in PostgreSQL.
- **Auditable changes:** retain review and decision history instead of overwriting the original model output.
- **Index deliberately:** index foreign keys and common filtering fields; validate index choices with query plans and representative volumes.

### 9.3 Conceptual entity relationship

```mermaid
erDiagram
    PRODUCT ||--o{ INSPECTION : inspected_in
    PRODUCT ||--o{ PRODUCT_VARIANT : has
    INSPECTION ||--o{ INSPECTION_IMAGE : uses
    INSPECTION ||--o{ INFERENCE_RUN : produces
    MODEL_VERSION ||--o{ INFERENCE_RUN : executes
    INFERENCE_RUN ||--o{ PREDICTION : generates
    INSPECTION ||--o{ INSPECTION_REVIEW : reviewed_by
    INSPECTION ||--o{ INSPECTION_EVENT : records
    DECISION_RULESET ||--o{ INSPECTION : evaluates
    EXPERIMENT_RUN ||--o{ EXPERIMENT_METRIC : reports
    MODEL_VERSION ||--o{ EXPERIMENT_RUN : evaluates
```

This is a conceptual starting point. The final physical ER diagram should be kept consistent with the actual migration scripts and implemented columns.

### 9.4 Suggested entities

| Entity | Responsibility | Typical fields |
|---|---|---|
| `product` | Product identity and traceability | `product_id`, `product_code`, `product_family`, `created_at` |
| `product_variant` | Variant-specific attributes | `variant_id`, `product_id`, `variant_code`, `attributes` |
| `inspection` | One inspection attempt and its final state | `inspection_id`, `product_id`, `status`, `started_at`, `completed_at`, `ruleset_id` |
| `inspection_image` | Input and derived-image metadata | `image_id`, `inspection_id`, `image_role`, `uri`, `sha256`, `width`, `height` |
| `model_version` | Immutable model release metadata | `model_version_id`, `model_name`, `version`, `artifact_uri`, `framework`, `created_at` |
| `inference_run` | One execution of a model or inspection task | `inference_run_id`, `inspection_id`, `model_version_id`, `status`, `latency_ms` |
| `prediction` | Structured result for a task or detected region | `prediction_id`, `inference_run_id`, `task_type`, `label`, `score`, `geometry` |
| `decision_ruleset` | Versioned acceptance rules | `ruleset_id`, `name`, `version`, `configuration` |
| `inspection_review` | Human review and disposition | `review_id`, `inspection_id`, `reviewer`, `decision`, `reason`, `reviewed_at` |
| `inspection_event` | Lifecycle/audit event | `event_id`, `inspection_id`, `event_type`, `event_time`, `details` |
| `experiment_run` | Reproducible training/evaluation execution | `experiment_id`, `model_version_id`, `dataset_version`, `parameters`, `started_at` |
| `experiment_metric` | Metrics from an experiment | `metric_id`, `experiment_id`, `metric_name`, `metric_value`, `scope` |

These are proposed names and field examples, not a substitute for the authoritative schema migration. The final schema may split or combine entities depending on the agreed requirements.

### 9.5 Data types and integrity

Recommended conventions:

- `UUID` or generated `BIGINT` primary keys, selected consistently.
- `TEXT` or bounded `VARCHAR` for names and codes where length constraints are meaningful.
- `TIMESTAMPTZ` for event timestamps.
- `BOOLEAN` for true/false flags.
- `NUMERIC` for values where exact decimal representation matters, such as specified tolerances; `DOUBLE PRECISION` may be appropriate for model scores and scientific metrics.
- `JSONB` for flexible, versioned configuration or auxiliary metadata—not as a replacement for core relational columns.
- `CHECK` constraints for valid ranges and controlled values.
- `UNIQUE` constraints for natural business identifiers where uniqueness is required.
- Foreign keys with explicit deletion behaviour; avoid cascading deletes that could erase inspection history unexpectedly.

### 9.6 Image and prediction artefacts

For each image, consider recording:

- Original/derived image role
- Storage URI or relative path
- SHA-256 checksum
- MIME type or file format
- Width, height, and channel count
- Capture timestamp and camera identifier where available
- Preprocessing configuration/version
- Retention class and access policy, if required

For predictions, consider recording task type, label, confidence/anomaly score, bounding box or mask reference, model version, threshold/ruleset version, and the final decision reason.

Large masks and anomaly maps can be stored as files or object-store artefacts, with metadata and references in PostgreSQL.

---

## 10. Proposed database schema

The following is a **logical schema outline** to guide implementation. It is not an executable migration and does not replace the project's reviewed SQL scripts.

```text
visioninspect
├── product
├── product_variant
├── inspection
├── inspection_image
├── model_version
├── inference_run
├── prediction
├── decision_ruleset
├── inspection_review
├── inspection_event
├── experiment_run
└── experiment_metric
```

### Suggested PostgreSQL namespaces

A simple first deployment may use the default `public` schema. A more structured deployment can separate responsibilities:

```text
visioninspect_core
  product, product_variant, inspection, inspection_image

visioninspect_ml
  model_version, inference_run, prediction, experiment_run, experiment_metric

visioninspect_audit
  inspection_event, inspection_review

visioninspect_config
  decision_ruleset
```

Schema-per-domain is optional. It introduces additional permissions and migration complexity, so it should be adopted only if it improves access control or maintainability for the actual deployment.

### Example table relationship rules

- An inspection must refer to a valid product when product-level traceability is required.
- An inference run must identify the inspection and model version used.
- A prediction must identify the inference run that generated it.
- A completed inspection should have a defined final disposition or a documented exception state.
- A review should reference the inspection and preserve reviewer, timestamp, reason, and decision.
- A model version should be immutable once used for production-like inference; corrections should create a new version.
- A ruleset version used for a decision should remain identifiable after the rule configuration changes.

### Migration and deployment recommendation

Keep SQL migrations under version control, for example:

```text
database/
├── migrations/
│   ├── 001_create_core_tables.sql
│   ├── 002_create_ml_tables.sql
│   ├── 003_add_constraints_and_indexes.sql
│   └── 004_seed_reference_data.sql
├── seeds/
└── README.md
```

Use migrations as the source of truth. Avoid manually changing the database schema in pgAdmin without recording the change in a migration. Apply migrations to a disposable development database first, then validate constraints, indexes, and rollback/recovery procedures.

---

## 11. Repository structure

The following structure is the **proposed target structure**. Directories and files should be marked implemented only when they exist and are tested in the repository.

```text
industrial-vision-inspection/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── pyproject.toml                 # or requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── docs/
│   ├── architecture/
│   ├── database/
│   │   ├── er-diagram/
│   │   ├── data-dictionary/
│   │   └── migrations-guide.md
│   ├── machine-vision/
│   ├── hardware/
│   ├── experiments/
│   └── decisions/
│
├── configs/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   └── decision_rules/
│
├── data/
│   ├── raw/                       # ignored by Git
│   ├── interim/                   # ignored by Git
│   ├── processed/                 # ignored by Git
│   └── README.md
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_baseline_experiment.ipynb
│   └── 03_error_analysis.ipynb
│
├── src/
│   └── visioninspect/
│       ├── __init__.py
│       ├── config/
│       ├── data/
│       │   ├── validation.py
│       │   └── dataset.py
│       ├── preprocessing/
│       ├── models/
│       │   ├── anomaly_detection/
│       │   └── screw_detection/
│       ├── colour/
│       ├── evaluation/
│       ├── decision/
│       ├── database/
│       │   ├── models.py
│       │   ├── repositories.py
│       │   └── session.py
│       ├── api/
│       │   └── main.py
│       └── services/
│
├── dashboard/
│   └── app.py
│
├── database/
│   ├── migrations/
│   ├── seeds/
│   └── README.md
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── api/
│   └── fixtures/
│
├── scripts/
│   ├── validate_dataset.py
│   ├── train_baseline.py
│   ├── evaluate_model.py
│   └── seed_demo_data.py
│
├── artifacts/                     # ignored by Git
└── reports/
    ├── evaluation/
    └── error_analysis/
```

### Repository hygiene

- Do not commit private, proprietary, or production images without authorisation.
- Keep large datasets and model weights out of Git unless there is a deliberate, approved versioning strategy.
- Commit configuration templates, not credentials.
- Use small, deterministic fixtures for tests.
- Document commands and expected outputs.
- Keep the README status aligned with actual repository state.

---

## 12. Components and responsibilities

| Component | Proposed technology | Responsibility |
|---|---|---|
| Image reading and transforms | OpenCV, Pillow | Load and transform images safely |
| Dataset validation | Python, pandas, Pillow/OpenCV | Check files, labels, masks, dimensions, and splits |
| Anomaly-detection baseline | PyTorch plus a suitable implementation/library | Train or fit the selected baseline and produce anomaly maps/scores |
| Evaluation | scikit-learn and task-specific metric code | Compute metrics and error analysis |
| Colour measurement | OpenCV and a colour-science library if needed | Calibrated colour conversion and tolerance evaluation |
| API | FastAPI | Accept inspection requests and return structured responses |
| Request validation | Pydantic | Validate API payloads and configuration |
| Relational database | PostgreSQL | Store traceability and structured results |
| Database access | SQLAlchemy and a PostgreSQL driver | Map entities and manage database operations |
| Dashboard | Streamlit | Display inspection results and experiment summaries |
| Automated tests | pytest | Validate unit, integration, and API behaviour |
| Containerisation | Docker / Docker Compose | Repeatable local service setup |
| Experiment tracking | MLflow, optional | Track runs, parameters, metrics, and artefacts |
| Logging | Python `logging` or structured logging package | Diagnostics and operational events |

The exact package list and versions should be pinned after the first working environment is tested.

---

## 13. Hardware architecture

### 13.1 Development workstation

A practical initial setup is a standard Windows, Linux, or macOS development machine. A GPU is helpful for some deep-learning experiments but is not mandatory for every CPU-capable baseline.

| Resource | Practical starting point | Notes |
|---|---|---|
| CPU | Modern 4–8 core processor | More cores can improve data preparation and concurrent service execution |
| RAM | 16 GB minimum practical target; 32 GB preferred for larger experiments | Dataset caching and multiple services increase memory demand |
| Storage | SSD with enough free space for dataset, environments, model artefacts, and logs | Keep raw data separate from generated artefacts |
| GPU | Optional for initial experiments; NVIDIA CUDA-compatible GPU recommended for accelerated PyTorch workloads | Check the selected model/library's VRAM requirements |
| Display | Standard development monitor | No special display hardware required |
| Network | Internet for initial dependency/dataset download | Offline deployment needs a dependency and artefact distribution plan |

These are planning recommendations, not verified requirements for every model. Actual needs depend on image resolution, batch size, backbone, feature memory, and whether training or only inference is being run.

### 13.2 Proposed industrial inspection cell

```text
[Product Positioning / Fixture]
              |
              v
[Presence / Trigger Sensor] ---> [PLC or Cell Controller]
              |                          |
              v                          v
[Industrial Camera + Lens]       [Trigger / Handshake]
              |
              v
[Controlled Lighting + Enclosure]
              |
              v
[Industrial PC / Edge Computer]
  - Image validation
  - Inference and decision rules
  - Local result buffer
  - API / database client
              |
              v
[PostgreSQL / Plant Network / Dashboard]
              |
              v
[PLC Decision Interface / Operator Review]
```

### 13.3 Camera, optics, and lighting

Selection must be based on the smallest defect that must be detected, field of view, working distance, product speed, surface finish, and environmental conditions.

Consider:

- **Camera type:** industrial area-scan or line-scan camera depending on motion and field of view.
- **Resolution:** sufficient pixels on the smallest defect of interest; calculate this from field of view and required spatial resolution rather than choosing megapixels in isolation.
- **Lens:** appropriate focal length, working distance, distortion, depth of field, and sensor format.
- **Lighting:** diffuse, dome, bar, coaxial, dark-field, or other lighting selected through experiments on the actual surface.
- **Enclosure:** reduce ambient-light variation and protect equipment where necessary.
- **Calibration:** geometric calibration and colour calibration when the measurement requires them.
- **Triggering:** photoelectric/proximity sensor or PLC trigger, with deterministic camera acquisition where required.

Shiny painted metal, curved surfaces, edges, and recesses may require multiple lighting directions or camera views. A software model cannot recover visual information that was never captured.

### 13.4 Edge compute and industrial integration

A future industrial PC should be selected after benchmarking the chosen model at target image size and expected line rate. Check CPU/GPU support, available memory, thermal conditions, operating-system compatibility, network interfaces, and recovery behaviour.

PLC integration may use a site-approved industrial protocol such as OPC UA or a vendor-supported interface. This is a future integration boundary, not a claim that a PLC connection currently exists. Safety-related machine control must remain within the appropriate engineered and validated control system; the AI prototype should not be treated as a safety controller.

### 13.5 Deployment stages

1. **Offline development:** stored benchmark images and local experiments.
2. **Software integration:** API, database, and dashboard running locally or in containers.
3. **Bench validation:** actual camera and controlled lighting, without controlling production equipment.
4. **Shadow-mode pilot:** system observes products and records results while the existing approved process remains authoritative.
5. **Controlled production integration:** only after acceptance criteria, failure handling, rollback, and responsible engineering approvals are satisfied.

---

## 14. Software stack and requirements

### 14.1 Proposed baseline environment

| Software | Purpose | Status |
|---|---|---|
| Python 3.11+ | Application and ML development | Proposed |
| Git | Version control | Recommended |
| PostgreSQL 16+ | Relational persistence | Proposed |
| pgAdmin 4 | Optional graphical database administration | Optional |
| PyTorch | Deep-learning model execution | Proposed |
| OpenCV | Image processing | Proposed |
| FastAPI | REST API | Proposed |
| Streamlit | Prototype dashboard | Proposed |
| pytest | Automated testing | Proposed |
| Docker Desktop / Docker Engine | Repeatable local services | Optional but recommended |
| JupyterLab / Jupyter Notebook | Exploration and experiments | Optional |
| VS Code / PyCharm | IDE | Developer choice |

Version numbers are compatibility targets to confirm during implementation. Use a tested, pinned dependency set rather than assuming that every latest release works with every model implementation.

### 14.2 Operating system

Supported development options may include:

- Windows 10/11 (64-bit)
- Ubuntu Linux LTS
- macOS, subject to the selected ML library and acceleration requirements

For a Windows developer workstation, a local Python virtual environment is sufficient for initial experiments. Docker Desktop can simplify running PostgreSQL and the API together. GPU acceleration requires a compatible driver and framework build.

### 14.3 Minimum software requirements

- Python and a package manager (`pip` or an equivalent).
- Git.
- A supported PyTorch build if a PyTorch model is selected.
- OpenCV/Pillow for image loading and validation.
- PostgreSQL for integrated persistence; it is not required for a standalone model experiment.
- A database driver and migration approach for schema management.
- Tests and documented configuration for repeatable execution.

### 14.4 Optional development tools

- pgAdmin 4 for inspecting PostgreSQL objects and data.
- Docker Compose for local multi-service orchestration.
- MLflow for experiment tracking.
- pre-commit, Ruff, or equivalent tools for linting and formatting.
- GitHub Actions or another CI system for automated test execution.

---

## 15. Python libraries

The following list is a **candidate dependency inventory**, not a declaration that every library is already installed or required. Add a dependency only when the code uses it.

| Library / package | Purpose | Priority |
|---|---|---|
| `numpy` | Numerical arrays and calculations | Core |
| `pandas` | Tabular metadata and evaluation summaries | Core for tabular workflows |
| `Pillow` | Image opening and format checks | Core or alternative to OpenCV for some tasks |
| `opencv-python` | Image transformations, filtering, colour conversion, and image I/O | Core for vision workflows |
| `torch` | Tensor operations and deep-learning execution | Required for PyTorch models |
| `torchvision` | Image transforms and model utilities | Model-dependent |
| `scikit-learn` | Splitting, standard metrics, utilities | Core for evaluation utilities |
| `matplotlib` | Visualising images, curves, and error cases | Development |
| `seaborn` | Statistical plots, if desired | Optional |
| `fastapi` | HTTP API framework | Integration phase |
| `uvicorn[standard]` | ASGI server | Integration phase |
| `pydantic` | Request/configuration validation | API phase |
| `sqlalchemy` | ORM and database session management | Database integration |
| `psycopg[binary]` or `psycopg2-binary` | PostgreSQL connectivity; choose one supported driver | Database integration |
| `alembic` | Versioned SQLAlchemy schema migrations | Recommended for ORM-managed migrations |
| `streamlit` | Prototype dashboard | Dashboard phase |
| `pytest` | Automated tests | Core quality tool |
| `httpx` | API testing client, if needed | Integration tests |
| `python-dotenv` | Local `.env` loading | Optional for local development |
| `pydantic-settings` | Typed settings from environment variables | Optional/configuration phase |
| `mlflow` | Experiment tracking and artefact metadata | Optional |
| `tqdm` | Progress bars for long-running scripts | Optional |
| `ruff` | Linting and formatting | Recommended |
| `pytest-cov` | Test coverage reporting | Recommended |

### 15.1 Candidate anomaly-detection implementations

PaDiM and PatchCore can be evaluated using a maintained implementation, such as an appropriate version of Anomalib, or a documented research implementation. Before adopting one, check:

- Supported Python and PyTorch versions
- Dataset loader and mask conventions
- Pretrained backbone availability and licensing
- GPU memory and inference requirements
- Metric definitions and threshold handling
- Export and deployment options

Do not add multiple competing frameworks to the initial environment until a baseline implementation has been selected.

### 15.2 Dependency management

For a reproducible project:

- Pin tested versions in `requirements.txt`, `pyproject.toml`, or a lock file.
- Record the Python version and operating system used for evaluation.
- Keep CPU and CUDA environment instructions clear if they differ.
- Separate production/runtime dependencies from notebook-only development dependencies when useful.
- Rebuild the environment periodically from scratch to confirm reproducibility.

---

## 16. Dataset strategy

### 16.1 Initial public benchmark

**MVTec Anomaly Detection (MVTec AD)** is the proposed initial benchmark for unsupervised industrial anomaly-detection experiments. The first category is `metal_nut`, which offers a bounded starting point for data validation, baseline fitting, and error analysis.

The public benchmark is useful for testing the software workflow; it does not represent the appearance distribution of the target appliance cabinets. Results on `metal_nut` must not be presented as validated cabinet-inspection performance.

### 16.2 Dataset preparation checklist

- Confirm dataset source, licence, and intended use.
- Inspect directory structure and image extensions.
- Count normal training images and normal/anomalous test images.
- Validate every image can be opened.
- Check image dimensions, colour channels, and corrupted files.
- Validate ground-truth mask correspondence and shape where masks exist.
- Ensure training and test data remain separated.
- Record dataset version, source, and preprocessing configuration.
- Keep dataset files out of Git unless explicitly approved.
- Save the validation report so later experiments use the same verified inputs.

### 16.3 Future production data

A production dataset should include representative variation across:

- Product families and variants
- Normal finish and acceptable manufacturing variation
- Defect categories and severity
- Camera views and production stations
- Lighting and exposure conditions
- Supplier/material/batch variation where relevant
- Product orientation, occlusion, and positioning
- Time periods and equipment maintenance states

Production data must be collected and labelled under an agreed process. Labels should be reviewed, and ambiguous examples should be tracked instead of silently treated as ground truth.

---

## 17. Evaluation and experimentation

### 17.1 Evaluation principles

- Keep a held-out test set separate from model and threshold selection.
- Use validation data to choose thresholds and tune parameters.
- Record model, data, preprocessing, and threshold versions.
- Report the metric's unit of analysis: image-level, pixel-level, defect-level, or inspection-level.
- Include error examples and the operational meaning of the result.
- Compare latency and resource usage as well as predictive performance.

### 17.2 Candidate metrics

| Metric | Meaning | Use |
|---|---|---|
| Precision | Of predicted positives, proportion that are truly positive | Understand false-alarm burden |
| Recall / sensitivity | Of true positives, proportion detected | Understand missed-defect risk |
| F1-score | Harmonic mean of precision and recall | Balanced summary where appropriate |
| Confusion matrix | Counts of correct and incorrect class decisions | Make error types explicit |
| ROC-AUC | Ranking quality across thresholds | Useful in some binary settings; may be misleading under strong imbalance |
| PR-AUC / Average Precision | Precision-recall trade-off | Often informative when defects are rare |
| Image-level AUROC | Separates normal and anomalous images by score | Image-level anomaly evaluation |
| Pixel-level AUROC | Separates anomalous and normal pixels | Localisation evaluation, when masks exist |
| Pixel-level Average Precision | Precision-recall quality at pixel level | Localisation under class imbalance |
| PRO / region-based metrics | Region-aware anomaly localisation evaluation | Optional, depending on benchmark/method |
| Inference latency | Time for a defined inference operation | Deployment planning |
| End-to-end latency | Intake through persisted result | System-level timing |
| Throughput | Completed inspections per unit time | Capacity planning |
| Resource usage | CPU, RAM, GPU memory and storage | Hardware planning |

Metric names and calculations must be documented precisely. For example, image-level AUROC and pixel-level AUROC measure different things and should never be reported as interchangeable.

### 17.3 Error analysis

Every evaluation report should ideally include:

- True-positive examples
- False-positive examples
- False-negative examples
- Normal examples with high anomaly scores
- Defects with low anomaly scores
- Anomaly maps or masks alongside source images
- Performance by defect type or relevant subgroup, where labels support it
- Threshold and parameter settings
- Known data limitations

### 17.4 Experiment record

For each run, save:

- Experiment ID and timestamp
- Dataset/version and split
- Git commit or code version
- Python/package versions
- Model and backbone
- Preprocessing configuration
- Hyperparameters and random seeds where applicable
- Threshold selection procedure
- Evaluation metrics
- Hardware and elapsed time
- Output artefact paths
- Notes on failure cases and next steps

---

## 18. Synthetic data and baseline plan

### 18.1 Why use synthetic data?

Synthetic records and simulated inspection responses can help validate the database, API, dashboard, and end-to-end data flow before actual production integration. Synthetic images can also help test image-processing utilities and explore defect augmentation.

### 18.2 What synthetic data can and cannot prove

Synthetic data can help test:

- Database constraints and relationships
- API payload validation
- Result persistence and retrieval
- Dashboard filters and status handling
- Error handling and idempotency
- Basic pipeline orchestration

Synthetic data alone cannot establish real-world defect-detection performance. Artificial scratches, colour changes, or screw images may not represent real material reflections, surface texture, defect morphology, optics, or manufacturing variation. Use synthetic data as a separate, clearly labelled experiment and evaluate the final baseline on a suitable held-out real dataset.

### 18.3 Recommended baseline sequence

1. Validate the public benchmark and generate a data-quality report.
2. Implement a simple, reproducible baseline with fixed preprocessing.
3. Record image-level metrics and pixel-level metrics where applicable.
4. Inspect anomaly maps and false-positive/false-negative cases.
5. Compare one alternative model only after the first baseline is reproducible.
6. Tune thresholds on validation data and report the selected operating point.
7. Add database persistence and API integration.
8. Use synthetic inspection records to exercise end-to-end system behaviour.
9. Replace simulated inputs with authorised production images when available.
10. Document what has been tested and what remains unvalidated.

---

## 19. Installation and local setup

The commands below are a starting template. Exact commands may change when dependency versions, packaging, and the implemented entry points are finalised.

### 19.1 Clone the repository

```bash
git clone https://github.com/samyak-anand/industrial-vision-inspection.git
cd industrial-vision-inspection
```

### 19.2 Create a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation due to execution policy, use an approved local policy or activate the environment from Command Prompt:

```bat
.venv\Scripts\activate.bat
```

**Linux/macOS**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 19.3 Install dependencies

If a tested dependency file is present:

```bash
pip install -r requirements.txt
```

If dependency management uses `pyproject.toml`, follow the repository's packaging instructions instead. Do not run both installation approaches blindly; use the method maintained by the project.

### 19.4 Configure environment variables

Create a local `.env` file from `.env.example` if that template exists. Keep real passwords and secret values out of Git.

Example values (placeholders only):

```dotenv
APP_ENV=development
LOG_LEVEL=INFO
DATABASE_URL=postgresql+psycopg://visioninspect_user:CHANGE_ME@localhost:5432/visioninspect
IMAGE_STORAGE_ROOT=./data
MODEL_ARTIFACT_ROOT=./artifacts
```

These variable names are proposed conventions. The application must be updated to read them before they are functional.

### 19.5 Create the PostgreSQL database

After installing PostgreSQL, create a development database and a least-privilege application role. Example SQL for a local development environment:

```sql
CREATE ROLE visioninspect_user
    LOGIN
    PASSWORD 'CHANGE_ME';

CREATE DATABASE visioninspect
    OWNER visioninspect_user;
```

Use a secure password and follow local PostgreSQL policy. Do not use a production password in a committed script. Run version-controlled migration files against the database using the migration procedure documented by the project. If the migrations are not yet implemented, the database schema is still a design specification rather than a runnable database.

### 19.6 Run tests

```bash
pytest
```

For coverage, if `pytest-cov` is installed:

```bash
pytest --cov=src --cov-report=term-missing
```

### 19.7 Run the API and dashboard

Once the corresponding entry points are implemented and configured, typical development commands may look like:

```bash
uvicorn visioninspect.api.main:app --reload
```

```bash
streamlit run dashboard/app.py
```

These commands depend on the final package layout and working entry points. They should not be considered verified execution commands until the application has been implemented and tested.

---

## 20. Configuration and environment variables

Configuration should be external to model code so experiments and deployment environments can be changed without editing source files.

| Variable / setting | Purpose | Example |
|---|---|---|
| `APP_ENV` | Runtime environment | `development` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `DATABASE_URL` | PostgreSQL connection string | Local DSN; secret kept outside Git |
| `IMAGE_STORAGE_ROOT` | Local image/artefact directory | `./data` |
| `MODEL_ARTIFACT_ROOT` | Model output directory | `./artifacts` |
| `MODEL_NAME` | Selected model | `patchcore-baseline` |
| `MODEL_VERSION` | Model version identifier | `0.1.0` |
| `CONFIG_PATH` | Configuration file location | `configs/models/baseline.yaml` |
| `INFERENCE_TIMEOUT_SECONDS` | API inference timeout | Set after measurement |
| `MAX_IMAGE_SIZE_MB` | Input file size guard | Set based on deployment requirements |

This is a proposed configuration contract. Use only variables that the implementation actually supports and document defaults, validation, and secret handling.

---

## 21. Testing and quality assurance

### Unit tests

- Image file validation and error paths
- Preprocessing output shape and data type
- Decision-rule threshold boundaries
- Prediction payload validation
- Identifier and status validation
- Metric calculation on known toy examples
- Database repository functions where they can be isolated

### Integration tests

- PostgreSQL migrations and constraints
- Inspection creation and result persistence
- Foreign-key relationships
- Model-version and inference-run traceability
- API request/response behaviour
- Database rollback and failure handling
- Dashboard data retrieval, where testable

### ML validation tests

- Dataset inventory and label/mask consistency
- No overlap between train and test splits
- Deterministic preprocessing checks
- Model output shape and finite-value checks
- Metric calculation on a small, known example
- Evaluation report completeness
- Regression checks against documented baselines, when stable and justified

### Acceptance checks before a pilot

- Agreed representative dataset and labelling procedure
- Defined false-acceptance and false-rejection criteria
- Verified image quality and camera/lighting repeatability
- Measured end-to-end latency under expected load
- Tested unavailable model, database outage, and malformed input scenarios
- Defined manual review, fallback, and recovery procedures
- Approved hardware, safety, network, retention, and integration design

---

## 22. Security, reliability and traceability

For a local proof of concept, implement proportionate controls early and strengthen them before any real production deployment.

- **Credentials:** keep passwords, API keys, and secrets in environment variables or a secret manager; never commit them.
- **Database privileges:** use an application role with only the permissions it needs.
- **Input validation:** enforce file type, size, dimensions, and request schema limits.
- **File safety:** use generated storage names and prevent untrusted paths from controlling file access.
- **Audit trail:** preserve model output and subsequent human review as separate records.
- **Retention:** define how long images, masks, logs, and inspection metadata should be retained.
- **Backups:** establish and test backup/restore procedures before relying on the database.
- **Failure handling:** distinguish a failed inspection from a technical error; do not silently convert an error into `PASS`.
- **Model governance:** retain version, artefact reference, training/evaluation metadata, and decision-rule configuration.
- **Monitoring:** measure latency, error rates, image-quality failures, score distributions, and review rates.
- **Production safety:** do not use an unvalidated prototype as the sole safety-critical or quality-release control.

---

## 23. Limitations and assumptions

1. The first benchmark experiment uses `metal_nut`, which is not a substitute for appliance-cabinet data.
2. Public datasets may not reflect production lighting, camera geometry, surface finishes, or defect distributions.
3. Synthetic database records validate software behaviour, not physical inspection accuracy.
4. Model metrics depend on dataset quality, split design, label reliability, and threshold selection.
5. Colour conformity requires a controlled imaging and calibration process.
6. Screw verification requires suitable labelled examples and clearly defined expected positions/types.
7. Camera, lens, lighting, fixture, trigger, and edge-compute specifications cannot be finalised without the actual product, smallest defect of interest, field of view, line speed, and environment.
8. Cycle-time requirements must be measured end-to-end on target hardware.
9. Database tables and API contracts may change during implementation; migrations and documentation must evolve together.
10. This README describes a proposed technical direction. It is not evidence of a commissioned industrial cell, validated model, or production deployment.

---

## 24. Development roadmap

| Stage | Deliverable | Completion evidence |
|---|---|---|
| 1. Requirements and design | Scope, architecture, data model, assumptions | Reviewed design and documented open questions |
| 2. Dataset audit | Dataset validator and data-quality report | Reproducible counts, integrity checks, and split validation |
| 3. Baseline ML | First anomaly-detection experiment | Saved configuration, code version, metrics, and error examples |
| 4. Error analysis and tuning | Threshold study and model comparison | Validation-based threshold selection and documented trade-offs |
| 5. PostgreSQL | Migrations and repository layer | Schema creation, integrity tests, sample records |
| 6. API integration | Inspection request and result endpoints | API tests and persisted inference results |
| 7. Dashboard | Inspection history and detail view | Demonstrable retrieval of saved results and evidence |
| 8. End-to-end tests | Integrated local workflow | Reproducible test run and documented failure handling |
| 9. Production-image study | Evaluation on authorised representative images | Data/label review and gap analysis against the benchmark |
| 10. Hardware feasibility | Camera/lighting/compute trial | Image-quality study and measured latency |
| 11. Pilot readiness review | Acceptance and deployment plan | Approved criteria, recovery strategy, and integration responsibilities |

A stage is complete only when the implementation and evidence exist. The roadmap is a plan, not a statement that these stages have already been delivered.

---

## 25. Definition of done

A feature is complete only when the following relevant items are satisfied:

- [ ] Code is committed and documented.
- [ ] Inputs, outputs, configuration, and failure modes are defined.
- [ ] Tests are added and pass.
- [ ] Data and model versions are recorded where applicable.
- [ ] Results are reproducible from documented commands.
- [ ] Metrics are calculated on the appropriate split and unit of analysis.
- [ ] False positives and false negatives are reviewed for ML changes.
- [ ] Database migrations and constraints are tested for schema changes.
- [ ] Security-sensitive values are not committed.
- [ ] README and technical documentation match the actual implementation.
- [ ] Remaining limitations and follow-up tasks are recorded.

---

## 26. References

Use the latest official documentation and verify version compatibility before implementation.

- **MVTec Anomaly Detection:** https://www.mvtec.com/company/research/datasets/mvtec-ad
- **PyTorch:** https://pytorch.org/docs/stable/
- **OpenCV:** https://docs.opencv.org/
- **PostgreSQL:** https://www.postgresql.org/docs/
- **pgAdmin:** https://www.pgadmin.org/docs/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Pydantic:** https://docs.pydantic.dev/
- **SQLAlchemy:** https://docs.sqlalchemy.org/
- **Alembic:** https://alembic.sqlalchemy.org/
- **scikit-learn:** https://scikit-learn.org/stable/
- **pytest:** https://docs.pytest.org/
- **Streamlit:** https://docs.streamlit.io/
- **Docker:** https://docs.docker.com/
- **MLflow:** https://mlflow.org/docs/latest/

---

## Maintainer note

Keep this README as the project's single high-level entry point. When implementation changes, update the status, commands, tested software versions, evaluation results, and limitations. Do not replace proposed architecture with claims of completed functionality unless the corresponding code has been executed and verified.
'''

