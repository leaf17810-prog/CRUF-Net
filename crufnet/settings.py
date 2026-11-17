import hashlib
import json
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('full', 'no_cru', 'no_ugf', 'baseline', 'fixed_blend', 'no_utility_loss')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            result.update(block)
    return result.hexdigest()


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def load_config(path, variant=None):
    config = read_json(path)
    required = ('dataset', 'protocol', 'train_split', 'validation_split', 'evaluation_split', 'epochs', 'scheduler_t_max', 'batch_size', 'seed', 'lr', 'weight_decay', 'eta_min', 'clip_norm', 'validation_every', 'warmup_epochs', 'ramp_epochs', 'loss_weights', 'utility', 'variant')
    missing = set(required) - set(config)
    if missing:
        raise ValueError('Missing configuration fields: ' + ', '.join(sorted(missing)))
    if variant is not None:
        config['variant'] = variant
    if config['variant'] not in VARIANTS:
        raise ValueError('Unknown model variant')
    if config['variant'] in ('no_ugf', 'baseline', 'fixed_blend', 'no_utility_loss'):
        config['utility']['weight'] = 0.0
    if config['dataset'] not in ('mus', 'deepsfp'):
        raise ValueError('Unknown dataset')
    if min(config['epochs'], config['scheduler_t_max'], config['batch_size'], config['validation_every'], config['ramp_epochs']) < 1:
        raise ValueError('Epochs, batch size, validation interval and ramp length must be positive')
    expected = {
        'mus': ('mus', 'train', 'val', 'val'),
        'deepsfp_development': ('deepsfp', 'development_train', 'development_val', 'development_val'),
        'deepsfp_final': ('deepsfp', 'official_train', None, 'official_test21'),
    }
    actual = tuple(config[k] for k in ('dataset', 'train_split', 'validation_split', 'evaluation_split'))
    if expected.get(config['protocol']) != actual:
        raise ValueError('Dataset split roles do not match the selected protocol')
    if config['protocol'] == 'deepsfp_final' and config['variant'] != 'full':
        raise ValueError('Ablation variants use the development protocol')
    if config['epochs'] > config['scheduler_t_max']:
        raise ValueError('Training cannot exceed the cosine schedule length')
    return config


def read_splits(dataset):
    manifest = read_json(ROOT / 'splits/manifest.json')
    result = {}
    for path in sorted((ROOT / 'splits' / dataset).glob('*.csv')):
        if digest_file(path) != manifest[path.relative_to(ROOT).as_posix()]:
            raise ValueError('Frozen split checksum mismatch: ' + path.name)
        lines = [line.strip() for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if lines[0].lower() == 'name':
            lines = lines[1:]
        if dataset == 'mus':
            lines = [name.removesuffix('.mat') for name in lines]
        if len(lines) != len(set(lines)):
            raise ValueError('Duplicate sample names')
        for name in lines:
            p = Path(name)
            if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name:
                raise ValueError('Invalid dataset-relative sample name')
        result[path.stem] = lines
    counts = {'train': 570, 'val': 84} if dataset == 'mus' else {'development_train': 212, 'development_val': 24, 'official_train': 236, 'official_test21': 21}
    if {key: len(value) for key, value in result.items()} != counts:
        raise ValueError('Frozen split counts differ from the paper protocol')
    pairs = [('train', 'val')] if dataset == 'mus' else [('development_train', 'development_val'), ('official_train', 'official_test21')]
    for first, second in pairs:
        a, b = set(result[first]), set(result[second])
        object_name = (lambda name: name.rsplit('_', 2)[0]) if dataset == 'mus' else (lambda name: Path(name).stem.rsplit('_', 1)[0])
        if a & b or {object_name(n) for n in a} & {object_name(n) for n in b}:
            raise ValueError('Object overlap across training and evaluation splits')
    if dataset == 'deepsfp' and set(result['development_train'] + result['development_val']) != set(result['official_train']):
        raise ValueError('Development splits do not partition the official training set')
    return result
