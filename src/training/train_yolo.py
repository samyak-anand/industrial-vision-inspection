"""Notebook cells 6-9: one-epoch smoke test, optional fine-tuning, results inspection."""
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train(*, mode='smoke', dataset=None, epochs=None, batch=2, imgsz=320):
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Use the configured GPU environment.')
    root = PROJECT_ROOT
    weights = root / 'weights' / 'yolo11n.pt'
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if mode not in {'smoke', 'baseline'}:
        raise ValueError('mode must be smoke or baseline')
    dataset = dataset or ('yolo_screw_hole' if mode == 'smoke' else 'yolo_screw_hole_grouped')
    yaml_path = root / 'data' / 'processed' / dataset / 'data.yaml'
    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_path)
    epochs = epochs if epochs is not None else (1 if mode == 'smoke' else 30)
    run_name = 'yolo11n_smoke_test' if mode == 'smoke' else 'yolo11n_grouped_baseline'
    print('GPU:', torch.cuda.get_device_name(0))
    print('Dataset:', yaml_path)
    print('Mode:', mode, 'Epochs:', epochs)
    model = YOLO(str(weights))
    model.train(
        data=str(yaml_path), epochs=epochs, imgsz=imgsz, batch=batch,
        device=0, workers=0, seed=42, amp=False,
        project=str(root / 'runs' / 'screw_hole'),
        name=run_name, exist_ok=False, save=True, val=True,
        **({'patience': 10, 'plots': True} if mode == 'baseline' else {}),
    )
    print('Training outputs:', model.trainer.save_dir)
    return Path(model.trainer.save_dir)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "baseline"), default="baseline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()
    train(mode=args.mode, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
