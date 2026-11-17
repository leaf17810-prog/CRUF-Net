import numpy as np

def _crop_indices(mask, threshold, crop=256):
    if min(mask.shape) < crop:
        raise ValueError('Native image smaller than crop')
    summed = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0, dtype=np.int64).cumsum(1, dtype=np.int64)
    counts = summed[crop:, crop:] - summed[:-crop, crop:] - summed[crop:, :-crop] + summed[:-crop, :-crop]
    indices = np.flatnonzero(counts > threshold * crop * crop).astype(np.int32)
    if not len(indices):
        raise ValueError('No eligible native crop')
    return (indices, counts.shape[1])

def viewing_encoding(top, left, height, width, sensor_height=1024, sensor_width=1224):
    u = (np.arange(left, left + width, dtype=np.float32) - sensor_width * 0.5) / (0.5 * sensor_width)
    v = (np.arange(top, top + height, dtype=np.float32) - sensor_height * 0.5) / (0.5 * sensor_height)
    uu, vv = np.meshgrid(u, v)
    return np.stack((uu, vv, np.ones_like(uu)))

def derived_fields(raw, mask):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[-1] != 4 or (not np.isfinite(raw).all()):
        raise ValueError('Expected finite HWC RAW4')
    q, u = (raw[..., 0] - raw[..., 2], raw[..., 1] - raw[..., 3])
    amplitude = np.hypot(q, u)
    cos2 = np.divide(q, amplitude, out=np.zeros_like(q), where=amplitude > 1e-08)
    sin2 = np.divide(u, amplitude, out=np.zeros_like(u), where=amplitude > 1e-08)
    mask = np.asarray(mask, dtype=np.float32)
    return np.stack((cos2 * mask, sin2 * mask, np.hypot(q, q + u), (raw.mean(-1) - amplitude) * mask, amplitude * mask), -1).astype(np.float32)
