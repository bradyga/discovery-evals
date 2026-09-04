"""Generate the pinned configuration for task w1-02.

The configuration is a free (non-periodic) binary cluster of coarse-grained
particles in vacuum.  Every particle carries `n_osc` classical harmonic
oscillator displacements.  Positions are drawn by rejection sampling inside a
sphere with a minimum separation, so the pair-distance distribution is broad
and the homo-coordination the model sees spans a wide range rather than
sitting on a plateau.

Alongside the pinned species labelling, the file carries two `demix_maps`:
alternative species labellings of the *same* positions, in which the two
species are spatially separated.  They are produced by sorting the positions
along a coordinate and by distance from the centroid -- position sorts and
nothing else.  This script computes no model quantity: no switching function,
no coordination number, no frequency, no energy.

Reduced Lennard-Jones units throughout: lengths in sigma_AA, energies in
eps_AA, oscillator displacements in sigma_AA.

Deterministic: numpy.random.default_rng(SEED), PCG64.  Re-running this script
reproduces config.json byte for byte.

    python make_config.py    # writes config.json next to this file
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 20260905
N_PARTICLES = 240
N_OSC = 3
RADIUS = 4.1  # cluster radius, sigma_AA
MIN_SEP = 0.95  # minimum pair separation, sigma_AA
XI_SIGMA = 0.30  # width of the oscillator-displacement distribution, sigma_AA
MAX_TRIES = 4_000_000

HERE = Path(__file__).resolve().parent


def sample_positions(rng: np.random.Generator) -> np.ndarray:
    """Rejection-sample N_PARTICLES inside a sphere with a hard minimum separation."""
    pos = np.empty((N_PARTICLES, 3), dtype=np.float64)
    n = 0
    tries = 0
    while n < N_PARTICLES:
        tries += 1
        if tries > MAX_TRIES:
            raise RuntimeError(f"failed to place {N_PARTICLES} particles in {MAX_TRIES} tries")
        cand = rng.uniform(-RADIUS, RADIUS, size=3)
        if cand @ cand > RADIUS**2:
            continue
        if n > 0:
            d = np.linalg.norm(pos[:n] - cand, axis=1)
            if d.min() < MIN_SEP:
                continue
        pos[n] = cand
        n += 1
    pos -= pos.mean(axis=0)
    return pos


def split_on(order: np.ndarray) -> np.ndarray:
    """Label the first half of `order` species 0 and the second half species 1."""
    labels = np.zeros(N_PARTICLES, dtype=np.int64)
    labels[order[N_PARTICLES // 2 :]] = 1
    return labels


def build() -> dict:
    rng = np.random.default_rng(SEED)
    positions = sample_positions(rng)
    species = np.zeros(N_PARTICLES, dtype=np.int64)
    species[N_PARTICLES // 2 :] = 1
    rng.shuffle(species)
    xi = rng.normal(0.0, XI_SIGMA, size=(N_PARTICLES, N_OSC))

    # Alternative labellings of the same positions, in which the species are
    # spatially separated.  Position sorts only.
    demix_maps = {
        "plane_x": split_on(np.argsort(positions[:, 0], kind="stable")),
        "radial": split_on(np.argsort(np.linalg.norm(positions, axis=1), kind="stable")),
    }

    return {
        "format": "w1-02/config/1",
        "units": "reduced Lennard-Jones (lengths in sigma_AA, energies in eps_AA)",
        "boundary": "none (free cluster in vacuum)",
        "seed": SEED,
        "generator": "make_config.py",
        "n_particles": N_PARTICLES,
        "n_osc": N_OSC,
        "species_encoding": {"0": "A", "1": "B"},
        "species": species.tolist(),
        "positions": positions.tolist(),
        "xi": xi.tolist(),
        "demix_maps_note": (
            "alternative species labellings of the same positions, produced by "
            "position sorts; see spec.md section 'The criterion'"
        ),
        "demix_maps": {k: v.tolist() for k, v in demix_maps.items()},
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = build()
    out = HERE / "config.json"
    out.write_text(json.dumps(cfg, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
