import argparse
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import NormalDataset
from .evaluation import evaluate
from .losses import training_loss
from .settings import load_config, read_json, write_json, digest_value, digest_file, VARIANTS
from .variants import build_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng():
    numpy_state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def save_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def optimizer_for(model, config):
    decay, exempt = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = exempt if getattr(parameter, '_no_weight_decay', False) or parameter.ndim <= 1 or name.endswith('.bias') else decay
            group.append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': config['weight_decay']}, {'params': exempt, 'weight_decay': 0.0}], lr=config['lr'])


def optimizer_step(model, batch, optimizer, config, device):
    inputs, target, mask = [batch[key].to(device) for key in ('x', 'gt', 'mask')]
    optimizer.zero_grad(set_to_none=True)
    loss, terms = training_loss(model(inputs), target, mask, config['loss_weights'], config['utility'])
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite training loss')
    loss.backward()
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_norm'], error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), terms, float(gradient), len(inputs)


def code_identity():
    root = Path(__file__).resolve().parent
    return digest_value({path.name: digest_file(path) for path in sorted(root.glob('*.py'))})


def train(config, cache, output, device, resume=False, stop_after_epoch=None, workers=0):
    output = Path(output).resolve()
    cache = Path(cache).resolve()
    if output == cache or cache in output.parents or output in cache.parents:
        raise ValueError('Training output and prepared cache must be separate directories')
    last_path = output / 'last.pt'
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError('Training output is not empty; use --resume or select a new output directory')
    if resume and not last_path.is_file():
        raise FileNotFoundError('Resume requires an existing last.pt')
    training_data = NormalDataset(cache, config, config['train_split'], epoch=1)
    validation_data = NormalDataset(cache, config, config['validation_split']) if config['validation_split'] else None
    data_identity = {name: training_data.manifest['records'][name] for name in training_data.names}
    if validation_data is not None:
        data_identity.update({name: validation_data.manifest['records'][name] for name in validation_data.names})
    signature = digest_value({'config': config, 'data': data_identity, 'source': code_identity()})
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config['seed'])
    model = build_model(config['variant']).to(device)
    optimizer = optimizer_for(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['scheduler_t_max'], eta_min=config['eta_min'])
    start, best_mae, best_epoch, history = 1, None, None, []
    if resume:
        state = torch.load(last_path, map_location=device, weights_only=True)
        if state['signature'] != signature:
            raise ValueError('Resume rejected: configuration, source code, or prepared data changed')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])
        start = state['epoch'] + 1
        best_mae, best_epoch, history = state['best_mae'], state['best_epoch'], state['history']
        if best_epoch == state['epoch']:
            save_checkpoint(output / 'best.pt', {'model': state['model'], 'config': config, 'epoch': best_epoch, 'validation_mae': best_mae})
        elif best_epoch is not None and not (output / 'best.pt').is_file():
            raise FileNotFoundError('Selected validation checkpoint is missing')
        del state
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', {'torch': str(torch.__version__), 'numpy': np.__version__, 'device': str(device), 'source_sha256': code_identity(), 'train_images': len(training_data), 'validation_images': len(validation_data) if validation_data is not None else 0})
    end = min(config['epochs'], stop_after_epoch) if stop_after_epoch is not None else config['epochs']
    if end < start - 1:
        raise ValueError('Stop epoch precedes the restored completed epoch')
    for epoch in range(start, end + 1):
        model.train()
        strength = max(0.0, min(1.0, (epoch - config['warmup_epochs']) / config['ramp_epochs']))
        model.set_refinement_strength(strength)
        training_data.epoch = epoch
        generator = torch.Generator().manual_seed(config['seed'] + epoch * 100003)
        loader = DataLoader(training_data, batch_size=config['batch_size'], shuffle=True, drop_last=False, num_workers=workers, pin_memory=device.type == 'cuda', generator=generator)
        count, total_loss, max_gradient = 0, 0.0, 0.0
        started = time.perf_counter()
        for index, batch in enumerate(loader, 1):
            loss, terms, gradient, samples = optimizer_step(model, batch, optimizer, config, device)
            count += samples
            total_loss += loss * samples
            max_gradient = max(max_gradient, gradient)
            if index == 1 or index % 5 == 0 or index == len(loader):
                print(f'Epoch {epoch}/{config["epochs"]} batch {index}/{len(loader)} loss={loss:.6f}', flush=True)
        scheduler.step()
        row = {'epoch': epoch, 'loss': total_loss / count, 'samples': count, 'optimizer_updates': len(loader),
               'lr': optimizer.param_groups[0]['lr'], 'refinement_strength': strength, 'gradient_norm_max': max_gradient,
               'train_seconds': time.perf_counter() - started}
        improved = False
        if validation_data is not None and (epoch % config['validation_every'] == 0 or epoch == config['epochs']):
            optimizer.zero_grad(set_to_none=True)
            result = evaluate(model, validation_data, device, output / 'validation' / f'epoch_{epoch:04d}')
            score = result['image_macro']['mae_deg']
            row['validation'] = result['image_macro']
            if best_mae is None or score < best_mae:
                best_mae, best_epoch, improved = score, epoch, True
        history.append(row)
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'config': config, 'signature': signature, 'epoch': epoch, 'best_mae': best_mae, 'best_epoch': best_epoch,
                 'history': history, 'rng': capture_rng()}
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(output / 'best.pt', {'model': state['model'], 'config': config, 'epoch': epoch, 'validation_mae': best_mae})
        write_json(output / 'history.json', history)
        write_json(output / 'status.json', {'completed_epochs': epoch, 'target_epochs': config['epochs'], 'best_epoch': best_epoch, 'best_validation_mae': best_mae})
        del state
    completed_epoch = history[-1]['epoch'] if history else 0
    if completed_epoch == config['epochs']:
        save_checkpoint(output / 'final.pt', {'model': model.state_dict(), 'config': config, 'epoch': completed_epoch})
    selected_checkpoint = 'best.pt' if validation_data is not None else 'final.pt'
    summary = {'training_complete': completed_epoch == config['epochs'], 'completed_epochs': completed_epoch,
               'selected_checkpoint': selected_checkpoint if (output / selected_checkpoint).is_file() else None, 'best_epoch': best_epoch,
               'selection': 'earliest minimum validation image-macro MAE' if validation_data is not None else 'fixed final epoch; test set not read during training'}
    write_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Train CRUF-Net from scratch with the frozen paper protocol')
    parser.add_argument('--config', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    args = parser.parse_args()
    config = load_config(args.config, args.variant)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; --device cpu is intended for small validation runs')
    if args.workers < 0:
        raise ValueError('Worker count cannot be negative')
    train(config, args.cache, args.output, device, args.resume, args.stop_after_epoch, args.workers)
