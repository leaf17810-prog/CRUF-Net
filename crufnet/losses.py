import math
import torch
import torch.nn.functional as F

def unit(x):
    return F.normalize(x, dim=1, eps=1e-06)

def normal_target(gt, mask, size):
    finite = torch.isfinite(gt).all(1, keepdim=True)
    if ((mask > 0.5) & ~finite).any():
        raise ValueError('Nonfinite foreground ground-truth normal')
    clean = torch.where((mask > 0.5) & finite, gt, torch.zeros_like(gt))
    valid = (mask > 0.5) & (clean.square().sum(1, keepdim=True) > 1e-12)
    normal = unit(clean)
    if tuple(size) == tuple(gt.shape[-2:]):
        return (normal, valid)
    coverage = F.interpolate(valid.float(), size=size, mode='area')
    average = F.interpolate(normal * valid, size=size, mode='area') / coverage.clamp_min(1e-06)
    usable = (coverage >= 0.5) & (average.square().sum(1, keepdim=True) > 1e-12)
    return (unit(average), usable)

def cosine_loss(pred, target, mask):
    if pred.shape != target.shape or mask.shape != pred[:, :1].shape:
        raise ValueError('Normal/mask shape mismatch')
    error = 1.0 - (unit(pred) * unit(target)).sum(1, keepdim=True).clamp(-1, 1)
    valid = mask.to(error.dtype)
    count = valid.sum((1, 2, 3))
    per_image = (error * valid).sum((1, 2, 3)) / count.clamp_min(1)
    usable = count > 0
    return per_image[usable].mean() if usable.any() else pred.sum() * 0.0

def angular_error_deg(pred, target):
    pred, target = (unit(pred.float()), unit(target.float()))
    cross = torch.linalg.cross(pred, target, dim=1).norm(dim=1, keepdim=True)
    dot = (pred * target).sum(1, keepdim=True).clamp(-1, 1)
    return torch.atan2(cross, dot) * (180.0 / math.pi)

@torch.no_grad()
def utility_targets(base, candidate, target, temperature_deg=1.0, margin_deg=0.25):
    if temperature_deg <= 0 or margin_deg <= 0:
        raise ValueError('Utility temperature and margin must be positive')
    gain = angular_error_deg(base, target) - angular_error_deg(candidate, target)
    preference = (gain / temperature_deg).sigmoid()
    importance = (gain.abs() / margin_deg).clamp_max(1)
    return (preference, importance, gain)

def utility_loss(refinement, target, valid, config):
    logits = refinement['logits']
    preference, importance, gain = utility_targets(refinement['base'], refinement['candidate'], target, temperature_deg=float(config.get('temperature_deg', 1.0)), margin_deg=float(config.get('margin_deg', 0.25)))
    if logits.shape != valid.shape or preference.shape != valid.shape:
        raise ValueError('Utility logit/target/mask shape mismatch')
    weight = importance * valid.to(importance.dtype)
    error = F.binary_cross_entropy_with_logits(logits.float(), preference, reduction='none')
    count = weight.sum((1, 2, 3))
    per_image = (error * weight).sum((1, 2, 3)) / count.clamp_min(1e-12)
    usable = count > 0
    value = per_image[usable].mean() if usable.any() else logits.sum() * 0.0
    with torch.no_grad():
        foreground = valid.bool()
        diagnostics = {'utility_gain_deg': gain[foreground].mean().item() if foreground.any() else 0.0, 'acceptance_mean': refinement['acceptance'][foreground].mean().item() if foreground.any() else 0.0, 'utility_importance_mean': importance[foreground].mean().item() if foreground.any() else 0.0}
    return (value, diagnostics)

def training_loss(outputs, gt, mask, weights=None, utility=None):
    weights = weights if weights is not None else {'normal': 1.0, 'half': 0.2, 'quarter': 0.1, 'candidate': 0.5}
    utility = utility if utility is not None else {'weight': 0.01, 'temperature_deg': 1.0, 'margin_deg': 0.25}
    predictions = {'normal': outputs['normal'], **outputs.get('aux_normals', {})}
    if 'refinement' in outputs:
        predictions['candidate'] = outputs['refinement']['candidate']
    terms = {}
    total = outputs['normal'].sum() * 0.0
    if not (mask > 0.5).flatten(1).any(1).all():
        raise ValueError('Training batch contains an empty foreground mask')
    for key, weight in weights.items():
        if key not in predictions:
            raise KeyError('Missing supervised output: ' + key)
        pred = predictions[key]
        target, valid = normal_target(gt, mask, pred.shape[-2:])
        value = cosine_loss(pred, target, valid)
        total = total + weight * value
        terms[key] = value.detach().item()
    if float(utility.get('weight', 0.01)) != 0:
        if 'refinement' not in outputs:
            raise KeyError('Missing utility-supervised refinement output')
        target, valid = normal_target(gt, mask, outputs['normal'].shape[-2:])
        value, diagnostics = utility_loss(outputs['refinement'], target, valid, utility)
        total = total + float(utility.get('weight', 0.01)) * value
        terms['utility'] = value.detach().item()
        terms.update(diagnostics)
    return (total, terms)
