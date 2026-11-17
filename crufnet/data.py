import argparse
import os
from pathlib import Path
import uuid
import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from .preprocessing import _crop_indices, derived_fields, viewing_encoding
from .settings import load_config, read_json, write_json, read_splits, digest_value

PNG_FILES = ('0.png', '45.png', '90.png', '135.png', 'mask.png', 'Normal_gt.png')


def record_folder(cache, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or ':' in name:
        raise ValueError('Invalid sample identifier')
    return Path(cache) / 'records' / path.with_suffix('')


def resolve_roots(dataset, mat_root=None, pol_root=None, data_root=None):
    if dataset == 'mus':
        if not mat_root or not pol_root:
            raise ValueError('MuS requires --mat-root and --pol-root')
        roots = {'mat': str(Path(mat_root).expanduser().resolve()), 'pol': str(Path(pol_root).expanduser().resolve())}
    else:
        if not data_root:
            raise ValueError('DeepSfP requires --data-root')
        base = Path(data_root).expanduser().resolve()
        if base.name == 'objects':
            base = base.parent
        elif (base / 'SurfaceNormals/objects').is_dir():
            base = base / 'SurfaceNormals'
        if not (base / 'objects').is_dir():
            raise FileNotFoundError('Expected SurfaceNormals/objects')
        roots = {'data': str(base)}
    for root in roots.values():
        if not Path(root).is_dir():
            raise FileNotFoundError(root)
    return roots


def source_signature(dataset, roots, name):
    if dataset == 'mus':
        paths = [Path(roots['mat']) / (name + '.mat')] + [Path(roots['pol']) / name / file for file in PNG_FILES]
    else:
        paths = [Path(roots['data']) / 'objects' / name]
    return [{'file': str(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns} for path in paths]


def check_record(folder, signature=None):
    record = read_json(Path(folder) / 'record.json')
    if record['schema'] != 1 or (signature is not None and signature != record['source_signature']):
        raise ValueError('Existing prepared sample differs; choose a fresh cache directory')
    for key, info in record['arrays'].items():
        path = Path(folder) / (key + '.npy')
        if path.stat().st_size != info['bytes']:
            raise ValueError('Incomplete prepared sample: ' + str(path))
        value = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(value.shape) != info['shape'] or str(value.dtype) != info['dtype']:
            raise ValueError('Prepared array header differs: ' + str(path))
    return record


def prepare_record(dataset, roots, cache, name, training):
    signature = source_signature(dataset, roots, name)
    destination = record_folder(cache, name)
    if destination.exists():
        return check_record(destination, signature)
    if dataset == 'mus':
        fields = loadmat(Path(roots['mat']) / (name + '.mat'), variable_names=['images', 'mask', 'I_Normal_gt', 'P', 'IrawD', 'IrawS'])
        folder = Path(roots['pol']) / name
        def image(file, mode):
            with Image.open(folder / file) as value:
                return np.array(value.convert(mode))
        raw = np.stack([image(file, 'L') for file in PNG_FILES[:4]], axis=-1).astype(np.float32) / 255
        mask = image('mask.png', 'L') > 127
        encoded = image('Normal_gt.png', 'RGB')
        if not np.array_equal(raw, fields['images']) or not np.array_equal(mask, fields['mask'] > 0) or not np.array_equal(encoded, fields['I_Normal_gt']):
            raise ValueError('MuS MAT/PNG mismatch: ' + name)
        if fields['P'].shape != (*mask.shape, 5):
            raise ValueError('MuS P must have five supplied channels')
        extra = np.concatenate((fields['P'][..., 1:4], fields['IrawD'][..., None], fields['IrawS'][..., None]), axis=-1)
        if extra.dtype != np.float32 or not np.isfinite(extra).all():
            raise ValueError('Invalid supplied MuS auxiliary fields')
        arrays = {'extra': extra}
        sensor = [*mask.shape, 0]
    else:
        fields = loadmat(Path(roots['data']) / 'objects' / name, variable_names=['images', 'mask', 'normals_gt'])
        raw = np.asarray(fields['images'])
        mask = np.asarray(fields['mask'])
        normal = np.asarray(fields['normals_gt'], dtype=np.float64)
        if raw.shape != (1024, 1224, 4) or normal.shape != (1024, 1224, 3) or mask.shape != (1024, 1224):
            raise ValueError('Unexpected DeepSfP source dimensions')
        if not np.isin(mask, (0, 1)).all() or not np.isfinite(raw).all() or not np.isfinite(normal).all():
            raise ValueError('Invalid DeepSfP observations or target')
        foreground = mask.astype(bool)
        length = np.linalg.norm(normal, axis=-1)
        if not foreground.any() or (length[foreground] <= 1e-6).any():
            raise ValueError('Empty mask or zero foreground normal')
        normal = np.divide(normal, length[..., None], out=np.zeros_like(normal), where=foreground[..., None])
        mask = mask[:, 100:1124].astype(np.uint8)
        arrays = {'raw4': np.ascontiguousarray(raw[:, 100:1124], dtype=np.float32),
                  'normal_gt': np.ascontiguousarray(normal[:, 100:1124], dtype=np.float32), 'mask': np.ascontiguousarray(mask)}
        sensor = [1024, 1224, 100]
    if not mask.any() or not np.isfinite(raw).all():
        raise ValueError('Empty or invalid source sample')
    threshold = 0.5 if dataset == 'mus' else 0.4
    if training:
        indices, width = _crop_indices(mask, threshold)
        arrays['crop_indices'] = indices
    else:
        width = mask.shape[1] - 255
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name('.' + destination.name + '.tmp_' + uuid.uuid4().hex)
    temporary.mkdir()
    info = {}
    for key, array in arrays.items():
        path = temporary / (key + '.npy')
        np.save(path, array, allow_pickle=False)
        info[key] = {'shape': list(array.shape), 'dtype': str(array.dtype), 'bytes': path.stat().st_size}
    if signature != source_signature(dataset, roots, name):
        raise RuntimeError('Source files changed during preparation')
    record = {'schema': 1, 'name': name, 'dataset': dataset, 'shape': list(mask.shape), 'sensor': sensor,
              'crop_grid_width': width, 'arrays': info, 'source_signature': signature}
    write_json(temporary / 'record.json', record)
    os.rename(temporary, destination)
    return record


def prepare(config, roots, cache, include_test=False):
    cache = Path(cache).resolve()
    for root in roots.values():
        source = Path(root)
        if cache == source or source in cache.parents:
            raise ValueError('The prepared cache must be outside original data directories')
    splits = read_splits(config['dataset'])
    roles = [config['train_split']]
    if config['validation_split']:
        roles.append(config['validation_split'])
    if include_test:
        roles.append(config['evaluation_split'])
    names = list(dict.fromkeys(name for role in roles for name in splits[role]))
    trainable = set(splits['train'] if config['dataset'] == 'mus' else splits['official_train'])
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / 'manifest.json'
    manifest = {'schema': 1, 'dataset': config['dataset'], 'roots': roots, 'records': {}}
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest['dataset'] != config['dataset'] or manifest['roots'] != roots:
            raise ValueError('Existing cache belongs to different source roots or dataset')
    for index, name in enumerate(names, 1):
        record = prepare_record(config['dataset'], roots, cache, name, name in trainable)
        manifest['records'][name] = digest_value(record)
        write_json(manifest_path, manifest)
        print(f'Prepared {index}/{len(names)}: {name}', flush=True)
    return manifest


class NormalDataset(Dataset):
    def __init__(self, cache, config, split, epoch=0, names=None):
        self.cache = Path(cache).resolve()
        self.config, self.split, self.epoch = config, split, int(epoch)
        self.manifest = read_json(self.cache / 'manifest.json')
        if self.manifest['dataset'] != config['dataset']:
            raise ValueError('Cache dataset does not match configuration')
        splits = read_splits(config['dataset'])
        if split not in splits:
            raise ValueError('Unknown split')
        self.names = list(splits[split] if names is None else names)
        if not set(self.names) <= set(splits[split]):
            raise ValueError('Sample list is outside the selected split')
        self.training = split == config['train_split']
        self.records = {}
        for name in self.names:
            if name not in self.manifest['records']:
                raise FileNotFoundError('Sample has not been prepared: ' + name)
            record = check_record(record_folder(self.cache, name))
            if digest_value(record) != self.manifest['records'][name]:
                raise ValueError('Prepared sample manifest differs: ' + name)
            if config['dataset'] == 'mus' and record['source_signature'] != source_signature('mus', self.manifest['roots'], name):
                raise ValueError('MuS source files changed after preparation: ' + name)
            if self.training and 'crop_indices' not in record['arrays']:
                raise ValueError('Training crop indices are missing')
            self.records[name] = record

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        record = self.records[name]
        folder = record_folder(self.cache, name)
        height, width = record['shape']
        top = left = 0
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([self.config['seed'], self.epoch, index]))
            indices = np.load(folder / 'crop_indices.npy', mmap_mode='r', allow_pickle=False)
            top, left = divmod(int(indices[int(rng.integers(len(indices)))]), record['crop_grid_width'])
            height = width = 256
        def field(key):
            values = np.load(folder / (key + '.npy'), mmap_mode='r', allow_pickle=False)
            return np.array(values[top:top + height, left:left + width], copy=True)
        if self.config['dataset'] == 'mus':
            source = Path(self.manifest['roots']['pol']) / name
            box = (left, top, left + width, top + height)
            def image(file, mode):
                with Image.open(source / file) as value:
                    return np.array(value.convert(mode).crop(box))
            raw = np.stack([image(file, 'L') for file in PNG_FILES[:4]], axis=-1).astype(np.float32) / 255
            mask = (image('mask.png', 'L') > 127).astype(np.float32)
            normal = image('Normal_gt.png', 'RGB').astype(np.float32) / 255 * 2 - 1
            physical = field('extra')
        else:
            raw, mask, normal = field('raw4'), field('mask').astype(np.float32), field('normal_gt')
            physical = derived_fields(raw, mask)
        sensor_h, sensor_w, offset_x = record['sensor']
        coordinates = viewing_encoding(top, left + offset_x, height, width, sensor_h, sensor_w)
        lengths = np.linalg.norm(normal, axis=-1)
        if not mask.any() or (lengths[mask > 0] <= 1e-6).any():
            raise ValueError('Invalid target normals')
        if self.config['protocol'] != 'deepsfp_final':
            normal = normal / np.maximum(lengths[..., None], 1e-8)
        normal = normal * mask[..., None]
        inputs = np.concatenate((raw.transpose(2, 0, 1), physical.transpose(2, 0, 1), coordinates), axis=0)
        if inputs.shape != (12, height, width) or not np.isfinite(inputs).all() or not np.isfinite(normal).all():
            raise ValueError('Invalid input tensor or target')
        if self.training and mask.mean() <= (0.5 if self.config['dataset'] == 'mus' else 0.4):
            raise ValueError('Training crop does not satisfy the foreground threshold')
        def tensor(array):
            return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        return {'x': tensor(inputs), 'gt': tensor(normal.transpose(2, 0, 1)), 'mask': tensor(mask[None]), 'name': name}


def main():
    parser = argparse.ArgumentParser(description='Prepare native CRUF-Net data and immutable splits')
    parser.add_argument('--config', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--mat-root')
    parser.add_argument('--pol-root')
    parser.add_argument('--data-root')
    parser.add_argument('--include-test', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    roots = resolve_roots(config['dataset'], args.mat_root, args.pol_root, args.data_root)
    prepare(config, roots, args.cache, args.include_test)
