"""Translation registration and height errors on the reference DSM grid."""

import numpy as np
from numba import njit


def overlap(shape, dx, dy):
    height, width = shape
    reference = np.s_[max(0, -dy) : min(height, height - dy), max(0, -dx) : min(width, width - dx)]
    predicted = np.s_[max(0, dy) : min(height, height + dy), max(0, dx) : min(width, width + dx)]
    return reference, predicted


@njit
def correlation(reference, predicted):
    count = 0
    sum_a = sum_b = sum_aa = sum_bb = sum_ab = 0.0
    for row in range(reference.shape[0]):
        for col in range(reference.shape[1]):
            a = reference[row, col]
            b = predicted[row, col]
            if np.isfinite(a) and np.isfinite(b):
                count += 1
                sum_a += a
                sum_b += b
                sum_aa += a * a
                sum_bb += b * b
                sum_ab += a * b
    if count < 2:
        return -np.inf
    var_a = sum_aa - sum_a * sum_a / count
    var_b = sum_bb - sum_b * sum_b / count
    if var_a <= 0 or var_b <= 0:
        return -np.inf
    return (sum_ab - sum_a * sum_b / count) / np.sqrt(var_a * var_b)


def downsample(raster):
    height, width = raster.shape
    padded = np.pad(raster, ((0, height % 2), (0, width % 2)), constant_values=np.nan)
    blocks = padded.reshape((height + 1) // 2, 2, (width + 1) // 2, 2)
    counts = np.isfinite(blocks).sum(axis=(1, 3))
    return np.divide(
        np.nansum(blocks, axis=(1, 3)), counts, out=np.full(counts.shape, np.nan), where=counts > 0
    )


def find_translation(reference, predicted, radius=5):
    x, y = 0, 0
    if min(reference.shape) > 100:
        x, y = find_translation(downsample(reference), downsample(predicted), radius)
        x, y = 2 * x, 2 * y

    best = (x, y)
    ref, pred = overlap(reference.shape, x, y)
    best_score = correlation(reference[ref], predicted[pred])
    for dy in range(max(1 - reference.shape[0], y - radius), min(reference.shape[0], y + radius + 1)):
        for dx in range(max(1 - reference.shape[1], x - radius), min(reference.shape[1], x + radius + 1)):
            ref, pred = overlap(reference.shape, dx, dy)
            score = correlation(reference[ref], predicted[pred])
            if score > best_score:
                best, best_score = (dx, dy), score
    return best


def evaluate(reference, predicted, classes=None):
    """Estimate an integer XY shift and a constant height offset, then score the overlap."""
    if reference.shape != predicted.shape:
        raise ValueError("Reference and predicted DSMs must share a grid.")
    dx, dy = find_translation(reference, predicted)
    ref, pred = overlap(reference.shape, dx, dy)
    delta = reference[ref] - predicted[pred]
    valid = np.isfinite(delta)
    dz = float(np.median(delta[valid])) if valid.any() else None
    registered = np.full(reference.shape, np.nan, dtype=np.float32)
    if dz is not None:
        registered[ref] = predicted[pred] + dz
    error = registered - reference
    absolute_error = np.abs(error)
    valid = np.isfinite(error)

    def mean(mask):
        return float(absolute_error[mask].mean()) if mask.any() else None

    metrics = {
        "mae_reg_all_m": mean(valid),
        "registration_dx_px": dx,
        "registration_dy_px": dy,
        "registration_dz_m": dz,
        "valid_pixels": int(valid.sum()),
    }
    if classes is not None:
        metrics["mae_reg_buildings_m"] = mean(valid & (classes == 6))
    return metrics, registered, error
