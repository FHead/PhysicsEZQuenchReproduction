"""Fit the pp reference spectrum used by the Chapter 1 calculations."""

import argparse
import json
import os
from pathlib import Path

_CACHE = Path(__file__).resolve().parent / "Output/cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Input/hepdata_ins1673184_table4.json"
INITIAL = np.array([113.722, 2.720, 0.584])


def spectrum(p, parameters):
    """ATLAS pp spectrum form; the high-p tail factors are fixed."""
    a, b, c = parameters
    return (a / p) ** (b + c * np.log(p)) / (1 + np.exp((p - 700) / 150))


def read_data():
    table = json.loads(SOURCE.read_text())
    rows = table["values"][4:13]  # Nine bins from 100 to 800 GeV.
    edges = np.array([float(row["x"][0]["low"]) for row in rows] +
                     [float(rows[-1]["x"][0]["high"])])
    values = np.array([float(row["y"][0]["value"]) for row in rows])
    errors = [{e["label"]: e for e in row["y"][0]["errors"]} for row in rows]
    stat = np.array([float(e["stat"]["symerror"]) for e in errors])
    systematic = np.array([(float(e["sys"]["asymerror"]["plus"]) -
                            float(e["sys"]["asymerror"]["minus"])) / 2
                           for e in errors])
    return edges, values, stat, systematic


def binned_spectrum(parameters, edges):
    # The agreed reference convention samples the function every 1 GeV and
    # divides the weighted histogram by the measured bin widths.
    p = np.arange(100.0, 1200.0)
    counts = np.histogram(p, bins=edges, weights=spectrum(p, parameters))[0]
    return counts / np.diff(edges)


def fit(edges, values, stat, systematic, nuisance):
    def objective(theta):
        prediction = binned_spectrum(theta[:3], edges)
        shift = systematic * theta[3] if nuisance else 0
        residual = (values - prediction - shift) / stat
        penalty = theta[3] ** 2 if nuisance else 0
        return 0.5 * (residual @ residual + penalty)

    start = np.r_[INITIAL, 0.0] if nuisance else INITIAL
    result = minimize(objective, start, method="L-BFGS-B",
                      options={"maxfun": 1000})
    if not result.success:
        raise RuntimeError(result.message)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "Output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    edges, values, stat, systematic = read_data()
    simple = fit(edges, values, stat, systematic, False)
    baseline = fit(edges, values, stat, systematic, True)
    parameters = baseline.x[:3]
    result = {
        "source": "HEPData ins1673184, Table 4, rows 5-13",
        "model": "(A/p)^(B+C*ln(p))/(1+exp((p-700)/150))",
        "binning": "1 GeV samples; numpy.histogram includes the last edge",
        "parameters": parameters.tolist(),
        "systematic_nuisance": float(baseline.x[3]),
        "half_chi2": float(baseline.fun),
        "stat_only_parameters": simple.x.tolist(),
        "stat_only_half_chi2": float(simple.fun),
        "bin_edges_GeV": edges.tolist(),
    }
    (args.output / "pp_fit.json").write_text(json.dumps(result, indent=2) + "\n")

    centers = (edges[:-1] + edges[1:]) / 2
    dense = np.linspace(edges[0], edges[-1], 1000)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(centers, values, yerr=stat, fmt="ko", label="ATLAS pp")
    ax.plot(dense, spectrum(dense, parameters), label="fit")
    ax.set(xlabel=r"$p_T$ [GeV]", ylabel=r"$d^2\sigma/dp_Tdy$ [nb/GeV]")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "pp_fit.pdf")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
