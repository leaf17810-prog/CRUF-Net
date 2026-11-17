import numpy as np
import torch
from torch.nn import functional as F

def tile_starts(length, tile=256, stride=128):
    if length < tile or not 0 < stride <= tile:
        raise ValueError('Image smaller than tile or invalid stride')
    return list(dict.fromkeys(list(range(0, length - tile + 1, stride)) + [length - tile]))

@torch.no_grad()
def tiled_predict(model, x, device, *, return_stages=False, return_refinement=False, stop=None, tile_size=256, stride=128):
    if model.training or x.ndim != 3 or x.shape[0] != 12:
        raise ValueError('Expected eval-mode model and 12xHxW tensor')
    if tile_size != 256 or stride != 128:
        raise ValueError('This experiment freezes 256/128 native inference')
    _, h, w = x.shape
    coords = [(y, z) for y in tile_starts(h, tile_size, stride) for z in tile_starts(w, tile_size, stride)]
    v = torch.hann_window(tile_size, periodic=False, dtype=torch.float32)
    weight = (v[:, None] * v[None, :]).clamp_min(0.001)[None]
    total = torch.zeros(3, h, w, dtype=torch.float32)
    den = torch.zeros(1, h, w, dtype=torch.float32)
    stage_totals, expected_stages = ({}, None)
    refinement_totals, expected_refinement = ({}, None)
    for y, z in coords:
        if stop is not None:
            stop()
        output = model(x[:, y:y + tile_size, z:z + tile_size].unsqueeze(0).to(device))
        pred = output['normal'] if isinstance(output, dict) else output
        if pred.shape != (1, 3, tile_size, tile_size) or not torch.isfinite(pred).all():
            raise FloatingPointError('Invalid predicted normal tile')
        pred = F.normalize(pred.float(), dim=1, eps=1e-06)[0].cpu()
        total[:, y:y + tile_size, z:z + tile_size] += pred * weight
        den[:, y:y + tile_size, z:z + tile_size] += weight
        if return_stages and isinstance(output, dict):
            stages = output.get('aux_normals', {})
            if expected_stages is None:
                expected_stages = set(stages)
            if set(stages) != expected_stages or not set(stages).issubset({'half', 'quarter'}):
                raise ValueError('Unexpected auxiliary stages')
            for name, stage in stages.items():
                factor = {'half': 2, 'quarter': 4}[name]
                if stage.shape != (1, 3, tile_size // factor, tile_size // factor) or not torch.isfinite(stage).all():
                    raise FloatingPointError('Invalid auxiliary normal tile')
                stage = F.normalize(F.interpolate(stage.float(), size=(tile_size, tile_size), mode='bilinear', align_corners=False), dim=1, eps=1e-06)[0].cpu()
                if name not in stage_totals:
                    stage_totals[name] = torch.zeros_like(total)
                stage_totals[name][:, y:y + tile_size, z:z + tile_size] += stage * weight
        if return_refinement and isinstance(output, dict):
            refinement = output.get('refinement', {})
            names = set(refinement).intersection({'base', 'candidate', 'acceptance'})
            if expected_refinement is None:
                expected_refinement = names
            if names != expected_refinement or (names and names != {'base', 'candidate', 'acceptance'}):
                raise ValueError('Incomplete or changing refinement diagnostic outputs')
            for name in names:
                channels = 1 if name == 'acceptance' else 3
                value = refinement[name]
                if value.shape != (1, channels, tile_size, tile_size) or not torch.isfinite(value).all():
                    raise FloatingPointError('Invalid refinement diagnostic tile: ' + name)
                if name == 'acceptance':
                    if ((value < 0) | (value > 1)).any():
                        raise ValueError('Refinement acceptance must lie in [0,1]')
                    value = value.float()[0].cpu()
                else:
                    value = F.normalize(value.float(), dim=1, eps=1e-06)[0].cpu()
                if name not in refinement_totals:
                    refinement_totals[name] = torch.zeros(channels, h, w, dtype=torch.float32)
                refinement_totals[name][:, y:y + tile_size, z:z + tile_size] += value * weight
    if not torch.all(den > 0):
        raise RuntimeError('Uncovered image pixels')
    pred = F.normalize(total / den, dim=0, eps=1e-06)
    stages = {k: F.normalize(v / den, dim=0, eps=1e-06) for k, v in stage_totals.items()}
    if return_refinement:
        refinement = {k: (v / den).clamp(0, 1) if k == 'acceptance' else F.normalize(v / den, dim=0, eps=1e-06) for k, v in refinement_totals.items()}
        return (pred, stages, refinement)
    if return_stages:
        return (pred, stages)
    return pred

def refinement_metrics(base, candidate, final, acceptance, gt, mask, margin_deg=0.25):
    if margin_deg != 0.25:
        raise ValueError('This experiment freezes the change diagnostic margin at 0.25 degrees')
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 3 and m.shape[0] == 1:
        m = m[0]
    gate = np.asarray(acceptance, dtype=np.float64)
    if gate.ndim == 3 and gate.shape[0] == 1:
        gate = gate[0]
    if gate.shape != m.shape or not m.any():
        raise ValueError('Invalid acceptance map or empty mask')
    a = gate[m]
    if not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
        raise FloatingPointError('Invalid foreground acceptance')
    bm, ba = angular_metrics(base, gt, m)
    cm, ca = angular_metrics(candidate, gt, m)
    fm, fa = angular_metrics(final, gt, m)
    candidate_gain, final_gain = (ba - ca, ba - fa)
    accurate = ba < 11.25

    def change(gain):
        improved, harmed = (gain > margin_deg, gain < -margin_deg)
        return dict(mean_gain_deg=float(gain.mean()), improved_gt_025_pct=float(improved.mean() * 100), harmed_gt_025_pct=float(harmed.mean() * 100), accurate_base_harmed_gt_025_pct_all=float((accurate & harmed).mean() * 100), accurate_base_harmed_gt_025_pct_conditional=float(harmed[accurate].mean() * 100) if accurate.any() else None)
    clear = np.abs(candidate_gain) > margin_deg
    pred_accept, useful = (a >= 0.5, candidate_gain > 0)
    gate_stats = dict(mean=float(a.mean()), p10=float(np.quantile(a, 0.1)), median=float(np.median(a)), p90=float(np.quantile(a, 0.9)), pct_below_01=float((a < 0.1).mean() * 100), pct_above_09=float((a > 0.9).mean() * 100), mean_on_useful_candidate=float(a[candidate_gain > margin_deg].mean()) if (candidate_gain > margin_deg).any() else None, mean_on_harmful_candidate=float(a[candidate_gain < -margin_deg].mean()) if (candidate_gain < -margin_deg).any() else None, decision_agreement_on_clear_candidate_pct=float((pred_accept[clear] == useful[clear]).mean() * 100) if clear.any() else None, clear_candidate_pixels=int(clear.sum()))
    if np.std(a) > 1e-12 and np.std(candidate_gain) > 1e-12:
        gate_stats['pearson_with_candidate_gain'] = float(np.corrcoef(a, candidate_gain)[0, 1])
    else:
        gate_stats['pearson_with_candidate_gain'] = None
    return dict(base=bm, candidate=cm, final=fm, candidate_vs_base=change(candidate_gain), final_vs_base=change(final_gain), final_vs_candidate_mean_gain_deg=float((ca - fa).mean()), accurate_base_pixels=int(accurate.sum()), foreground_pixels=int(m.sum()), acceptance=gate_stats, oracle_endpoint_gt_only=dict(mae_deg=float(np.minimum(ba, ca).mean()), scope='Uses GT to choose the better independently fused base/candidate endpoint at each pixel; diagnostic only, not model output or guaranteed bound on final blending'), definition=dict(change_margin_deg=margin_deg, accurate_base_threshold_deg=11.25, gate='Hann-weighted tile acceptance; no vector normalization', fusion='Base, candidate and final normals independently unit-normalized and Hann-merged; averaged gate cannot reproduce final normals from fused endpoints'))

def metrics_from_angles(angles):
    a = np.asarray(angles, dtype=np.float64)
    if not a.size or not np.isfinite(a).all():
        raise FloatingPointError('Empty or nonfinite angular errors')
    return dict(mae_deg=float(a.mean()), median_deg=float(np.median(a)), rmse_deg=float(np.sqrt(np.mean(a * a))), acc_11_25_pct=float(np.mean(a < 11.25) * 100), acc_22_5_pct=float(np.mean(a < 22.5) * 100), acc_30_pct=float(np.mean(a < 30) * 100))

def angular_metrics(pred, gt, mask):
    pred, gt = (np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64))
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if pred.shape != gt.shape or pred.shape != (3, *mask.shape) or (not mask.any()):
        raise ValueError('Incompatible normal/mask dimensions or empty mask')
    if not np.isfinite(pred[:, mask]).all() or not np.isfinite(gt[:, mask]).all():
        raise FloatingPointError('Nonfinite foreground normal')
    pn, gn = (np.linalg.norm(pred, axis=0), np.linalg.norm(gt, axis=0))
    if (gn[mask] <= 1e-06).any():
        raise ValueError('Zero-length foreground GT')
    dot = (pred * gt).sum(0) / (np.maximum(pn, 1e-06) * np.maximum(gn, 1e-06))
    angles = np.degrees(np.arccos(np.clip(dot[mask], -1, 1)))
    return (metrics_from_angles(angles), angles)
