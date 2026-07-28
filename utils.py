import os
import numpy as np
import torch
import PIL.Image
import matplotlib.pylab as pl


# NumPy image utilities

def to_rgb(x):
    """Render NCHW CA states to NHWC RGB using the same semantics as CLIPLoss."""
    from config import CLIP_RENDER_MODE, STATE_CLIP_VALUE

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.nan_to_num(x, nan=0.0, posinf=STATE_CLIP_VALUE, neginf=-STATE_CLIP_VALUE)
    x = np.clip(x, -STATE_CLIP_VALUE, STATE_CLIP_VALUE)
    x = x.transpose(0, 2, 3, 1)

    if CLIP_RENDER_MODE == 'hard':
        rgb = np.clip(x[..., :3], 0.0, 1.0)
        alpha = np.clip(x[..., 3:4], 0.0, 1.0)
    else:
        rgb = 1.0 / (1.0 + np.exp(-x[..., :3]))
        alpha = 1.0 / (1.0 + np.exp(-5.0 * (x[..., 3:4] - 0.5)))

    return np.clip(1.0 - alpha + alpha * rgb, 0.0, 1.0)


def np2pil(a):
    if a.dtype in [np.float32, np.float64]:
        a = np.uint8(np.clip(a, 0, 1) * 255)
    return PIL.Image.fromarray(a)

def imwrite(path, a, fmt=None):
    a = np.asarray(a)
    if fmt is None:
        fmt = path.rsplit('.', 1)[-1].lower()
        if fmt == 'jpg':
            fmt = 'jpeg'
    np2pil(a).save(path, fmt, quality=95)

def tile2d(a, w=None):
    a = np.asarray(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w - len(a)) % w
    a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), 'constant')
    h = len(a) // w
    a = a.reshape([h, w] + list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))
    return a

# Training visualization

def generate_pool_figures(pool, step_i, log_dir='train_log', rank_fn=None, highlight_first=False, prefix=''):
    """
    Visualize states from a pattern pool.

    Args:
        pool: SamplePool instance.
        step_i: Current training step.
        log_dir: Output directory.
        rank_fn: Optional function returning indices from best to worst.
        highlight_first: Draw a red border around the first ranked sample.
        prefix: Optional filename prefix.
    """
    if rank_fn is not None:
        indices = rank_fn(pool.x)
        top49 = pool.x[indices[:49]]
    else:
        top49 = pool.x[:49]

    imgs = to_rgb(top49)  # [49, H, W, 3]
    if highlight_first and len(imgs) > 0:
        b = 2  # Border width in pixels.
        imgs[0, :b,  :,  :] = [1.0, 0.0, 0.0]
        imgs[0, -b:, :,  :] = [1.0, 0.0, 0.0]
        imgs[0, :,  :b,  :] = [1.0, 0.0, 0.0]
        imgs[0, :, -b:,  :] = [1.0, 0.0, 0.0]
    tiled = tile2d(imgs)
    imwrite(os.path.join(log_dir, '%s%04d_pool.jpg' % (prefix, step_i)), tiled)

def visualize_batch(x0, x, step_i, log_dir='train_log', prefix=''):
    vis0 = np.hstack(to_rgb(x0))
    vis1 = np.hstack(to_rgb(x))
    vis = np.vstack([vis0, vis1])
    imwrite(os.path.join(log_dir, '%sbatches_%04d.jpg' % (prefix, step_i)), vis)
    print('batch saved to', os.path.join(log_dir, 'batches_%04d.jpg' % step_i))

def plot_loss(loss_log, save_path=None):
    pl.figure(figsize=(10, 4))
    pl.title('CLIP loss history')
    pl.plot(loss_log, '.', alpha=0.1)
    pl.axhline(0.0, color='black', linewidth=0.5, alpha=0.3)
    if save_path:
        pl.savefig(save_path, dpi=120)
        print('Saved loss curve to', save_path)
    pl.close()


# Damage masks

def make_circle_masks(n, h, w):
    """Generate random circular NCHW masks for damage training."""
    x = np.linspace(-1.0, 1.0, w)[None, None, None, :]   # [1,1,1,W]
    y = np.linspace(-1.0, 1.0, h)[None, None, :, None]   # [1,1,H,1]
    cx = np.random.uniform(-0.5, 0.5, (n, 1, 1, 1))
    cy = np.random.uniform(-0.5, 0.5, (n, 1, 1, 1))
    r  = np.random.uniform(0.1, 0.4,  (n, 1, 1, 1))
    return ((x - cx)**2 / r**2 + (y - cy)**2 / r**2 < 1.0).astype(np.float32)


# --- Sample Pool --------------------------------------------------

class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            setattr(self, k, np.asarray(v))

    def sample(self, n, probabilities=None):
        idx = np.random.choice(self._size, n, False, p=probabilities)
        return self.take(idx)

    def take(self, indices):
        idx = np.asarray(indices, dtype=np.int64)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        return SamplePool(**batch, _parent=self, _parent_idx=idx)

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)
