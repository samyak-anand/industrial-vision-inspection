"""Notebook cells 1-5: inspect pretrained checkpoint and YOLO dataset YAML."""
from pathlib import Path
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def inspect_model(dataset='yolo_screw_hole_grouped'):
    from ultralytics import YOLO
    root = PROJECT_ROOT
    checkpoint = root / 'weights' / 'yolo11n.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    yaml_path = root / "data" / "processed" / dataset / "data.yaml"
    with yaml_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = YOLO(str(checkpoint))
    print('Checkpoint:', checkpoint)
    print('Task:', model.task)
    print('Pretrained class count:', len(model.names))
    print('First five classes:', list(model.names.items())[:5])
    print('Dataset configuration:', config)
    head = model.model.model[-1]
    print('Detection head:', type(head).__name__)
    print('Pretrained detection classes:', head.nc)
    print('Box regression channels:', head.reg_max * 4)
    print('Feature scales:', len(head.stride))
    print('Feature strides:', head.stride.tolist())
    model.info(detailed=True)
    print('Note: pretrained head classes differ from the two-class fine-tuning dataset.')


if __name__ == "__main__":
    inspect_model()
