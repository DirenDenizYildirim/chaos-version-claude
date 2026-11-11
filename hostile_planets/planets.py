"""Planet presets: per-blob amplitudes (type-major, K slots per type) and ambient corrosion."""

import numpy as np

from .constants import K, NB, RHO_AMB

PLANETS = ("corrosion", "jam", "quake", "combined")

# rows: corrosion, jam, quake amplitudes for the K slots of that type
_AMPS = {
    "corrosion": [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "jam":       [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
    "quake":     [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    "combined":  [[0.8, 0.8, 0.0], [0.8, 0.8, 0.0], [0.8, 0.8, 0.0]],
    "benign":    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
}
_AMBIENT = {"corrosion": RHO_AMB, "jam": 0.0, "quake": 0.0, "combined": RHO_AMB, "benign": 0.0}


def planet_amp(planet: str) -> np.ndarray:
    """Amplitude vector [NB] for a planet preset."""
    amp = np.asarray(_AMPS[planet], dtype=np.float32)
    assert amp.shape == (3, K)
    return amp.reshape(NB)


def planet_ambient(planet: str) -> float:
    return float(_AMBIENT[planet])
