"""VisionInspect AI: validate YOLO screw/hole data, baseline weights, and GPU.

Run from any working directory:
    python validate_yolo_baseline.py
    python validate_yolo_baseline.py --project-root PATH

Exit codes: 0 = data/model passed; 1 = data/model failed; 2 = setup failure.
GPU status is reported separately; use --require-gpu to make GPU failure fatal.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SPLITS = ('train', 'val', 'test')
EXPECTED_CLASSES = {0: 'screw', 1: 'hole'}
EXPECTED_WEIGHTS_SHA256 = '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1'


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_yaml(path: Path):
    import yaml
    with path.open(encoding='utf-8-sig') as f:
        return yaml.safe_load(f)


def dataset_check(dataset_root: Path, errors: list[str], warnings: list[str]):
    from PIL import Image
    yaml_path = dataset_root / 'data.yaml'
    result = {'yaml': str(yaml_path), 'splits': {}, 'total_images': 0,
              'total_annotations': 0, 'exact_cross_split_duplicate_groups': [],
              'repeated_stems_across_splits': {}}
    if not yaml_path.is_file():
        errors.append(f'Missing YAML: {yaml_path}')
        return result
    try:
        cfg = read_yaml(yaml_path)
    except Exception as exc:
        errors.append(f'Cannot read YAML: {exc}')
        return result
    if not isinstance(cfg, dict):
        errors.append('YAML must be a mapping')
        return result
    names = cfg.get('names')
    if isinstance(names, list):
        names = dict(enumerate(names))
    if names != EXPECTED_CLASSES:
        errors.append(f'Expected classes {EXPECTED_CLASSES}; found {names!r}')
    if cfg.get('nc') is not None and cfg['nc'] != 2:
        errors.append(f"YAML nc must be 2, got {cfg['nc']!r}")
    raw_root = cfg.get('path')
    if not isinstance(raw_root, str) or not raw_root.strip():
        errors.append('YAML path must be a nonempty string')
        return result
    declared_root = Path(raw_root)
    if not declared_root.is_absolute():
        declared_root = (dataset_root / declared_root).resolve()
    if declared_root.resolve() != dataset_root.resolve():
        errors.append(f'YAML path {declared_root} does not match dataset root {dataset_root}')
    hashes = defaultdict(list)
    stems = defaultdict(set)
    for split in SPLITS:
        relative = cfg.get(split)
        if not isinstance(relative, str) or not relative.strip():
            errors.append(f'Missing or invalid YAML {split} path')
            continue
        image_dir = declared_root / relative
        label_dir = declared_root / 'labels' / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f'Missing {split} directories: images={image_dir.is_dir()}, labels={label_dir.is_dir()}')
            continue
        images = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob('*.txt'))
        image_map = defaultdict(list)
        label_map = defaultdict(list)
        counts = Counter()
        for p in images:
            image_map[p.stem].append(p)
            stems[p.stem].add(split)
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im.load()
                hashes[digest_file(p)].append((split, p.name))
            except Exception as exc:
                errors.append(f'Unreadable image {p}: {exc}')
        for p in labels:
            label_map[p.stem].append(p)
        for stem, paths in image_map.items():
            if len(paths) != 1:
                errors.append(f'{split}: multiple images with stem {stem}')
        for stem, paths in label_map.items():
            if len(paths) != 1:
                errors.append(f'{split}: multiple labels with stem {stem}')
        for stem in sorted(image_map.keys() - label_map.keys()):
            errors.append(f'{split}: image missing label: {stem}')
        for stem in sorted(label_map.keys() - image_map.keys()):
            errors.append(f'{split}: orphan label: {stem}')
        for label in labels:
            try:
                lines = label.read_text(encoding='utf-8-sig').splitlines()
            except Exception as exc:
                errors.append(f'Cannot read label {label}: {exc}')
                continue
            for n, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f'{label}:{n}: expected five values')
                    continue
                try:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    errors.append(f'{label}:{n}: invalid class or coordinates')
                    continue
                if cls not in EXPECTED_CLASSES:
                    errors.append(f'{label}:{n}: unknown class {cls}')
                    continue
                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    errors.append(f'{label}:{n}: non-finite coordinate')
                    continue
                eps = 1e-6
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    errors.append(f'{label}:{n}: invalid normalized box')
                    continue
                if x-w/2 < -eps or x+w/2 > 1+eps or y-h/2 < -eps or y+h/2 > 1+eps:
                    errors.append(f'{label}:{n}: bounding box outside image')
                    continue
                counts[cls] += 1
        result['splits'][split] = {'images': len(images), 'label_files': len(labels),
                                   'screw': counts[0], 'hole': counts[1],
                                   'annotations': sum(counts.values())}
        result['total_images'] += len(images)
        result['total_annotations'] += sum(counts.values())
    duplicate_groups = [entries for entries in hashes.values() if len({s for s, _ in entries}) > 1]
    result['exact_cross_split_duplicate_groups'] = duplicate_groups
    if duplicate_groups:
        errors.append(f'{len(duplicate_groups)} exact duplicate image groups across splits')
    repeated = {stem: sorted(groups) for stem, groups in stems.items() if len(groups) > 1}
    result['repeated_stems_across_splits'] = repeated
    if repeated:
        warnings.append(f'{len(repeated)} image stems repeat across splits; review for leakage')
    warnings.append('Perceptual near-duplicates and same-part/group leakage are not tested')
    return result


def model_check(weights: Path, errors: list[str]):
    result = {'path': str(weights), 'exists': weights.is_file(), 'sha256': None,
              'task': None, 'pretrained_classes': None, 'loaded': False}
    if not weights.is_file():
        errors.append(f'Missing model weights: {weights}')
        return result
    try:
        result['sha256'] = digest_file(weights)
        if result['sha256'] != EXPECTED_WEIGHTS_SHA256:
            errors.append(f'Checkpoint hash mismatch: {result["sha256"]}')
            return result
        from ultralytics import YOLO
        model = YOLO(str(weights))
        result['task'] = model.task
        result['pretrained_classes'] = len(model.names)
        result['loaded'] = True
        if model.task != 'detect' or len(model.names) != 80:
            errors.append(f'Unexpected pretrained model: task={model.task}, classes={len(model.names)}')
    except Exception as exc:
        errors.append(f'Checkpoint load failed: {exc}')
    return result


def gpu_check():
    result = {'nvidia_smi': None, 'pytorch_version': None, 'pytorch_cuda': None,
              'cuda_available': False, 'device_name': None, 'compute_capability': None,
              'vram_gib': None, 'compiled_architectures': [], 'compute_test': 'NOT RUN',
              'usable_for_pytorch': False, 'errors': []}
    try:
        proc = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True,
                              timeout=10, check=True)
        result['nvidia_smi'] = proc.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        result['errors'].append(f'nvidia-smi unavailable: {exc}')
    try:
        import torch
        result['pytorch_version'] = torch.__version__
        result['pytorch_cuda'] = torch.version.cuda
        result['cuda_available'] = bool(torch.cuda.is_available())
        if not result['cuda_available']:
            result['errors'].append('PyTorch CUDA unavailable (CPU-only wheel or incompatible CUDA setup)')
            return result
        result['device_name'] = torch.cuda.get_device_name(0)
        result['compute_capability'] = list(torch.cuda.get_device_capability(0))
        result['vram_gib'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        result['compiled_architectures'] = torch.cuda.get_arch_list()
        a = torch.randn(64, 64, device='cuda:0')
        b = a @ a
        torch.cuda.synchronize()
        if not bool(torch.isfinite(b).all().item()):
            raise RuntimeError('GPU result contains non-finite values')
        result['compute_test'] = 'PASSED'
        result['usable_for_pytorch'] = True
    except Exception as exc:
        result['compute_test'] = f'FAILED: {exc}'
        result['errors'].append(str(exc))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--require-gpu', action='store_true', help='Fail if CUDA computation does not pass')
    parser.add_argument('--skip-model', action='store_true', help='Skip Ultralytics checkpoint load')
    args = parser.parse_args()
    root = args.project_root.resolve()
    dataset_root = root / 'data' / 'processed' / 'yolo_screw_hole'
    errors, warnings = [], []
    try:
        dataset = dataset_check(dataset_root, errors, warnings)
        model = {'skipped': True} if args.skip_model else model_check(root / 'weights' / 'yolo11n.pt', errors)
        gpu = gpu_check()
    except ImportError as exc:
        print(f'Missing dependency: {exc}. Install pillow, pyyaml, ultralytics and torch.', file=sys.stderr)
        return 2
    if args.require_gpu and not gpu['usable_for_pytorch']:
        errors.append('GPU required but PyTorch CUDA computation did not pass')
    status = 'PASSED' if not errors else 'FAILED'
    report = {'project': 'VisionInspect AI', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'status': status, 'dataset': dataset, 'model': model, 'gpu': gpu,
              'errors': errors, 'warnings': warnings}
    report_path = dataset_root / 'reports' / 'yolo_baseline_readiness.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nVISIONINSPECT AI — YOLO BASELINE READINESS')
    print('=' * 54)
    for split, data in dataset['splits'].items():
        print(f"{split:5}: {data['images']} images, {data['label_files']} labels, "
              f"screw={data['screw']}, hole={data['hole']}")
    print(f"Total: {dataset['total_images']} images, {dataset['total_annotations']} annotations")
    print(f"Model: {'PASSED' if model.get('loaded') and not any('Checkpoint' in e or 'model weights' in e or 'pretrained model' in e for e in errors) else 'NOT VERIFIED'}")
    print(f"CUDA available: {gpu['cuda_available']}; GPU compute test: {gpu['compute_test']}")
    print(f'Dataset/model readiness: {status}; errors={len(errors)}; warnings={len(warnings)}')
    for msg in errors[:15]:
        print('ERROR:', msg)
    for msg in gpu['errors'][:5]:
        print('GPU NOTE:', msg)
    print('Report:', report_path)
    return 0 if not errors else 1


if __name__ == '__main__':
    sys.exit(main())
