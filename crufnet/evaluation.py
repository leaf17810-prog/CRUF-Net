import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from .data import NormalDataset
from .metrics import angular_metrics, tiled_predict
from .settings import load_config, write_json, VARIANTS
from .variants import build_model


@torch.no_grad()
def evaluate(model, dataset, device, output=None, save_predictions=False):
    model.eval()
    rows = []
    output = Path(output) if output is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    for index in range(len(dataset)):
        item = dataset[index]
        if save_predictions:
            prediction, _, refinement = tiled_predict(model, item['x'], device, return_refinement=True)
        else:
            prediction = tiled_predict(model, item['x'], device)
            refinement = {}
        values, _ = angular_metrics(prediction.numpy(), item['gt'].numpy(), item['mask'].numpy())
        rows.append({'name': item['name'], **values})
        print(f'Evaluate {index + 1}/{len(dataset)} {item["name"]}: MAE={values["mae_deg"]:.6f}', flush=True)
        if output is not None and save_predictions:
            path = output / 'predictions' / Path(item['name']).with_suffix('.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {'pred': prediction.numpy(), 'gt': item['gt'].numpy(), 'mask': item['mask'].numpy(), 'raw_mean': item['x'][:4].mean(0).numpy()}
            arrays.update({key: value.numpy() for key, value in refinement.items()})
            np.savez_compressed(path, **arrays)
    if not rows:
        raise ValueError('The evaluation split is empty')
    keys = [key for key in rows[0] if key != 'name']
    summary = {'images': len(rows), 'aggregation': 'arithmetic mean of per-image metrics',
               'image_macro': {key: float(np.mean([row[key] for row in rows])) for key in keys}, 'per_image': rows}
    if dataset.config['dataset'] == 'mus':
        levels = sorted({Path(row['name']).stem.rsplit('_', 1)[1] for row in rows})
        summary['by_turbidity'] = {}
        for level in levels:
            group = [row for row in rows if Path(row['name']).stem.rsplit('_', 1)[1] == level]
            summary['by_turbidity'][level] = {'images': len(group), **{key: float(np.mean([row[key] for row in group])) for key in keys}}
    if output is not None:
        write_json(output / 'metrics.json', summary)
        with (output / 'per_image_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate a locally trained CRUF-Net checkpoint')
    parser.add_argument('--config', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-predictions', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config, args.variant)
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        raise FileExistsError('Evaluation output is not empty; choose a new output directory')
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'config' not in state or 'model' not in state:
        raise ValueError('Expected a checkpoint produced by train.py')
    if state['config'] != config:
        raise ValueError('Checkpoint and requested configuration differ')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu for a small local check')
    model = build_model(config['variant'])
    model.load_state_dict(state['model'], strict=True)
    model.to(device)
    dataset = NormalDataset(args.cache, config, config['evaluation_split'])
    result = evaluate(model, dataset, device, args.output, args.save_predictions)
    write_json(Path(args.output) / 'evaluation_config.json', {'config': config, 'checkpoint_epoch': state['epoch'], 'split': config['evaluation_split']})
    print(result['image_macro'], flush=True)
