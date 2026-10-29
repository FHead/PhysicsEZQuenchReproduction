"""Reproduce the central-only deterministic-loss posterior (paper Figure 5)."""

import argparse
import json
import os
import time
from pathlib import Path

_CACHE = Path(__file__).resolve().parent / "Output/cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))

import emcee
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import differential_evolution


HERE = Path(__file__).resolve().parent
PAPER_MODES = {"diagonal": (1.41, 0.20), "covariant_log": (1.08, 0.26)}
SEEDS = {"diagonal": (20261112, 20261114),
         "covariant_log": (20261113, 20261115)}


def spectrum(p, parameters):
    a, b, c = parameters
    return (a / p) ** (b + c * np.log(p)) / (1 + np.exp((p - 700) / 150))


class Model:
    def __init__(self, kind, step):
        self.kind = kind
        parameters = json.loads((HERE / "Input/pp_fit.json").read_text())["parameters"]
        with h5py.File(HERE / "Input/ATLAS_data_public.h5") as source:
            self.edges = source["RAA_pedge"][...]
            self.observed = source["RAA_vals"][0, ...]
            stat = source["cov_RAA_stat_diag"][:10, :10]
            total = source["cov_RAA_sys_total"][:10, :10]
            sources = source["cov_RAA_sys_sum"][:10, :10]
        if kind == "diagonal":
            covariance = stat + np.diag(np.diag(total))
        else:
            covariance = (stat + sources) / np.outer(self.observed, self.observed)
        self.cholesky = np.linalg.cholesky(covariance)
        self.p = np.arange(150.0, 800.0, step)
        self.weights = spectrum(self.p, parameters)
        self.denominator = np.histogram(self.p, bins=self.edges, weights=self.weights)[0]
        if np.any(self.denominator <= 0):
            raise ValueError("empty pp reference bin")

    def predict(self, a, b):
        # Histogramming the shifted initial momenta includes the Jacobian
        # automatically: each initial quadrature weight is counted once.
        loss = a * self.p**b * np.log(self.p)
        shifted = np.histogram(self.p - loss, bins=self.edges, weights=self.weights)[0]
        return shifted / self.denominator

    def log_probability(self, theta):
        a, b = theta
        if not (0 < a < 10 and -1 < b < 10):
            return -np.inf
        prediction = self.predict(a, b)
        if self.kind == "covariant_log":
            if np.any(prediction <= 0):
                return -np.inf
            residual = np.log(prediction) - np.log(self.observed)
        else:
            residual = prediction - self.observed
        white = np.linalg.solve(self.cholesky, residual)
        return -0.5 * float(white @ white)


def run_chain(model, seed, steps, walkers, output):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # emcee's stretch move uses NumPy's global stream.
    optimum = differential_evolution(
        lambda x: -model.log_probability(x),
        bounds=[(0.01, 9.99), (-0.99, 1.0)], seed=seed,
        maxiter=100, popsize=15, polish=True,
    )
    initial = np.column_stack((
        np.clip(rng.normal(optimum.x[0], 0.25, walkers), 1e-5, 9.999),
        np.clip(rng.normal(optimum.x[1], 0.08, walkers), -0.999, 9.999),
    ))
    backend = emcee.backends.HDFBackend(output / f"{model.kind}_{seed}.h5")
    backend.reset(walkers, 2)
    sampler = emcee.EnsembleSampler(
        walkers, 2, model.log_probability, backend=backend,
        moves=[(emcee.moves.StretchMove(), 0.8), (emcee.moves.DEMove(), 0.2)],
    )
    started = time.perf_counter()
    sampler.run_mcmc(initial, steps, progress=False)
    try:
        tau = sampler.get_autocorr_time(tol=0)
        burn = max(500, int(5 * max(tau)))
    except Exception:
        tau = np.array([np.nan, np.nan])
        burn = steps // 4
    burn = min(burn, steps // 4) if burn >= steps // 2 else burn
    samples = backend.get_chain(discard=burn, flat=True)
    report = {
        "seed": seed, "steps": steps, "walkers": walkers,
        "optimized_mode": optimum.x.tolist(),
        "burn_steps": burn,
        "autocorrelation_steps": tau.tolist(),
        "mean_acceptance": float(np.mean(sampler.acceptance_fraction)),
        "median": np.median(samples, axis=0).tolist(),
        "seconds": time.perf_counter() - started,
    }
    (output / f"{model.kind}_{seed}.json").write_text(json.dumps(report, indent=2) + "\n")
    return samples, report


def contour(ax, samples, color):
    hist, xe, ye = np.histogram2d(samples[:, 0], samples[:, 1],
                                  bins=80, range=((0, 10), (-0.5, 0.5)))
    hist = gaussian_filter(hist, 1.2)
    ranked = np.sort(hist.ravel())[::-1]
    mass = np.cumsum(ranked) / ranked.sum()
    levels = sorted(ranked[np.searchsorted(mass, [0.95, 0.68])])
    ax.contour((xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2,
               hist.T, levels=levels, colors=color)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--repeat-steps", type=int, default=5000)
    parser.add_argument("--walkers", type=int, default=24)
    parser.add_argument("--step-GeV", type=float, default=0.03)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "Output")
    args = parser.parse_args()
    models = {kind: Model(kind, args.step_GeV) for kind in SEEDS}
    if args.check_only:
        print("zero-loss max |RAA-1| =",
              max(np.max(np.abs(model.predict(0, 0) - 1)) for model in models.values()))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    all_samples, reports = {}, {}
    for kind, seeds in SEEDS.items():
        pairs = [run_chain(models[kind], seed, steps, args.walkers, args.output)
                 for seed, steps in zip(seeds, (args.steps, args.repeat_steps))]
        all_samples[kind] = np.vstack([pair[0] for pair in pairs])
        reports[kind] = [pair[1] for pair in pairs]

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    for kind, color, target in (("diagonal", "tab:blue", axes[1, 0]),
                                ("covariant_log", "tab:red", axes[0, 1])):
        samples = all_samples[kind]
        axes[0, 0].hist(samples[:, 0], bins=60, range=(0, 10),
                        histtype="step", density=True, color=color, label=kind)
        axes[1, 1].hist(samples[:, 1], bins=60, range=(-0.5, 0.5),
                        histtype="step", density=True, color=color)
        contour(target, samples, color)
        target.plot(*PAPER_MODES[kind], "k*", markersize=10)
        target.set(xlabel="a", ylabel="b", xlim=(0, 10), ylim=(-0.5, 0.5),
                   title=kind.replace("_", " "))
    axes[0, 0].legend()
    axes[0, 0].set(xlabel="a", ylabel="density")
    axes[1, 1].set(xlabel="b", ylabel="density")
    fig.savefig(args.output / "figure5_posterior.pdf")
    plt.close(fig)
    (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
