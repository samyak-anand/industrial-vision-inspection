"""Inspect metrics and checkpoint files from a completed YOLO training run."""
from pathlib import Path
import pandas as pd

def inspect_run(run_dir):
    run_dir = Path(run_dir)
    for filename in ('weights/best.pt', 'weights/last.pt', 'results.csv', 'results.png', 'args.yaml'):
        print(f'{filename}: {"FOUND" if (run_dir / filename).is_file() else "MISSING"}')
    csv = run_dir / 'results.csv'
    if not csv.is_file():
        raise FileNotFoundError(csv)
    metrics = pd.read_csv(csv)
    metrics.columns = metrics.columns.str.strip()
    print('\nLast epoch metrics:')
    print(metrics.tail(1).T.to_string(header=False))
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    inspect_run(parser.parse_args().run_dir)
