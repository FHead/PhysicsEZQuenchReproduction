"""Generate Pb+Pb TRENTO profiles and the hard-jet sampling index."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
CENTRALITY = np.array([0.0, 0.1, 0.3, 0.5, 0.8])


def index_profiles(path):
    """Assign centrality by the agreed public ranking and compute W=sum rho²."""
    with h5py.File(path, "r") as source:
        # h5py's key order is the ordering used by the original index and
        # therefore preserves the random-number-to-event mapping downstream.
        names = list(source.keys())
        count = len(names)
        npart = np.empty(count)
        mult = np.empty(count)
        weights = np.empty(count)
        for i, name in enumerate(names):
            event = source[name]
            if event.shape != (100, 100):
                raise ValueError(f"unexpected grid shape in {name}: {event.shape}")
            rho = event[...]
            if not np.all(np.isfinite(rho)) or np.any(rho < 0):
                raise ValueError(f"invalid density in {name}")
            npart[i] = event.attrs["npart"]
            mult[i] = event.attrs["mult"]
            weights[i] = np.square(rho).sum()

    score = npart + mult / (np.max(mult) + 1.0)
    thresholds = np.quantile(score, 1.0 - CENTRALITY)
    labels = np.full(count, -1, dtype=np.int8)
    probabilities = np.zeros(count)
    for centrality, (upper, lower) in enumerate(zip(thresholds[:-1], thresholds[1:])):
        selected = (score <= upper) & (score > lower)
        labels[selected] = centrality
        probabilities[selected] = weights[selected] / weights[selected].sum()
    return names, npart, mult, score, weights, labels, probabilities, thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--trento", default=shutil.which("trento"))
    parser.add_argument("--output", type=Path, default=HERE / "Output")
    parser.add_argument("--reuse-profiles", action="store_true")
    args = parser.parse_args()
    if args.events < 10 or not args.trento:
        parser.error("need at least ten events and a TRENTO executable")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"PbPb_{args.events}_seed{args.seed}"
    h5_path = args.output / f"{stem}.h5"
    command = [
        args.trento, "Pb", "Pb", str(args.events), "--quiet",
        "--output", str(h5_path), "--reduced-thickness", "0",
        "--fluctuation", "1", "--nucleon-width", "1",
        "--nucleon-min-dist", "1.4", "--cross-section", "6.4",
        "--grid-max", "10", "--grid-step", "0.2",
        "--random-seed", str(args.seed),
    ]
    if args.reuse_profiles:
        if not h5_path.exists():
            parser.error(f"missing profiles for reuse: {h5_path}")
        elapsed = None
    else:
        if h5_path.exists():
            parser.error(f"{h5_path} exists; use --reuse-profiles to rebuild its index")
        start = time.perf_counter()
        # Keep TRENTO's cross-section cache inside this task's ignored output.
        environment = os.environ.copy()
        cache = args.output / "cache"
        cache.mkdir(exist_ok=True)
        environment["XDG_DATA_HOME"] = str(cache)
        subprocess.run(command, check=True, env=environment)
        elapsed = time.perf_counter() - start

    names, npart, mult, score, weights, labels, probability, thresholds = index_profiles(h5_path)
    if len(names) != args.events or np.any(weights <= 0):
        raise ValueError("event count or hard-jet weights are invalid")
    index_path = args.output / f"{stem}.index.npz"
    np.savez_compressed(index_path, event_names=names, npart=npart, mult=mult,
                        score=score, W=weights, class_label=labels,
                        event_probability_within_class=probability,
                        score_quantiles_high_to_low=thresholds)
    report = {
        "trento_version": subprocess.check_output([args.trento, "--version"], text=True).strip(),
        "command": command,
        "events": len(names), "grid_cells": [100, 100],
        "centrality_percent_edges": [0, 10, 30, 50, 80],
        "class_counts": [int(np.count_nonzero(labels == i)) for i in range(4)],
        "excluded_80_to_100_percent": int(np.count_nonzero(labels == -1)),
        "hard_jet_weight": "W = sum of squared cell densities",
        "centrality_score": "npart + mult/(max(mult)+1)",
        "generation_seconds": elapsed,
        "profiles_bytes": h5_path.stat().st_size,
    }
    (args.output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
