import numpy as np
from typing import Tuple


def make_synthetic_dataset(name: str, seed: int):
    """
    Dispatch to one of our four synthetic generators.
    """
    name = name.lower()
    if name == 'bump3':
        return make_bump3_dataset(seed=seed)
    elif name == 'sine_freq':
        return make_sine_freq_dataset(seed=seed)
    elif name == 'step_pos':
        return make_step_pos_dataset(seed=seed)
    elif name == 'square_duty':
        return make_square_duty_dataset(seed=seed)
    else:
        raise ValueError(f"Unknown synthetic dataset '{name}' ")


def make_sine_freq_dataset(
    samples_per_class=300,
    length=128,
    freqs=(2.0, 5.0, 8.0),
    noise_std=0.1,
    seed=0
):
    """
    Class k: sin(2π·freqs[k]·t) + Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length, endpoint=False)
    X, y = [], []
    for k, f in enumerate(freqs):
        wave = np.sin(2 * np.pi * f * t)
        for _ in range(samples_per_class):
            X.append(wave + rng.normal(0, noise_std, size=length))
            y.append(k)
    idx = rng.permutation(len(y))
    return np.array(X, dtype='float32')[idx], np.array(y, dtype='int64')[idx]


def make_step_pos_dataset(
    samples_per_class=300,
    length=128,
    step_positions=(32, 64, 96),
    noise_std=0.05,
    seed=0
):
    """
    Class k: series is 0 before pos[k], then 1 afterwards, plus noise.
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, pos in enumerate(step_positions):
        base = np.concatenate([np.zeros(pos), np.ones(length-pos)])
        for _ in range(samples_per_class):
            X.append(base + rng.normal(0, noise_std, size=length))
            y.append(k)
    idx = rng.permutation(len(y))
    return np.array(X, dtype='float32')[idx], np.array(y, dtype='int64')[idx]



def make_square_duty_dataset(
    samples_per_class=300,
    length=128,
    n_periods=4,
    duties=(0.25, 0.50, 0.75),
    noise_std=0.05,
    seed=0
):
    """
    Class k: square wave over n_periods with duty= duties[k].
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    period = length // n_periods
    for k, duty in enumerate(duties):
        high_len  = int(period * duty)
        low_len   = period - high_len
        prot = np.tile(np.concatenate([np.ones(high_len), np.zeros(low_len)]), n_periods)
        # pad if needed
        prot = np.pad(prot, (0, max(0, length-len(prot))), mode='wrap')[:length]
        for _ in range(samples_per_class):
            X.append(prot + rng.normal(0, noise_std, size=length))
            y.append(k)
    idx = rng.permutation(len(y))
    return np.array(X, dtype='float32')[idx], np.array(y, dtype='int64')[idx]



def make_bump3_dataset(
        samples_per_class=300, length=100,
        bump_locs=(30, 70), sigma=3.0,
        noise_std=0.1, amp_range=(0.8, 1.2),
        seed=0
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Class 0: one Gaussian at t=bump_locs[0]
    Class 1: one Gaussian at t=bump_locs[1]
    Class 2: sum of both bumps.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length)

    # build prototypes
    P0 = np.exp(-(t - bump_locs[0])**2 / (2*sigma**2))
    P1 = np.exp(-(t - bump_locs[1])**2 / (2*sigma**2))
    protos = [P0, P1, P0 + P1]

    X, y = [], []
    for cls, proto in enumerate(protos):
        for _ in range(samples_per_class):
            a = rng.uniform(*amp_range)
            noise = rng.normal(0, noise_std, size=length)
            X.append(a * proto + noise)
            y.append(cls)

    X = np.stack(X).astype("float32")
    y = np.array(y, dtype="int64")
    # shuffle
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

