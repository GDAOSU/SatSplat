import numpy as np
import matplotlib.pyplot as plt


def _normalize_img(img, perc=(0, 100), per_channel=True):
    """Normalize image to [0, 1] range based on percentile."""
    if per_channel:
        out = np.zeros_like(img)
        for c in range(img.shape[-1]):
            lo, hi = np.nanpercentile(img[..., c], perc)
            out[..., c] = (img[..., c] - lo) / (hi - lo)
        out = np.clip(out, 0, 1)
    else:
        lo, hi = np.nanpercentile(img, perc)
        out = (img - lo) / (hi - lo)
        out = np.clip(out, 0, 1)
    return out


def vizImg(img, perc=(2, 98), per_channel=True, skip_normalize=False, cmap="viridis"):
    if img.ndim == 3:
        if img.shape[-1] == 1:
            if skip_normalize:
                nimg = img
            else:
                nimg = _normalize_img(img, perc, per_channel=False)
            nimg = plt.get_cmap(cmap)(nimg[..., 0])[..., 0:3]

        elif img.shape[-1] == 3:
            if skip_normalize:
                nimg = img
            else:
                nimg = _normalize_img(img, perc, per_channel=per_channel)

        elif img.shape[-1] == 8:
            img = img[..., [4, 2, 1]]  # WorldView RGB band
            if skip_normalize:
                nimg = img
            else:
                nimg = _normalize_img(img, perc, per_channel=per_channel)
        else:
            if skip_normalize:
                nimg = img[..., 0]
            else:
                nimg = _normalize_img(img[..., 0], perc, per_channel=False)
            nimg = plt.get_cmap(cmap)(nimg[..., 0])[..., 0:3]

    else:
        if skip_normalize:
            nimg = img
        else:
            nimg = _normalize_img(img, perc, per_channel=False)
        nimg = plt.get_cmap(cmap)(nimg)[..., 0:3]

    return (nimg * 255).astype(np.uint8)
