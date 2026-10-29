"""Reproduce the central gamma-loss fit with an independent PCA+GP posterior."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

# Keep Poppler, Fontconfig, and Matplotlib caches inside ignored task output.
_CACHE = Path(__file__).resolve().parent / "Output/cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))

import emcee
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize
from scipy.special import gammainc
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


HERE = Path(__file__).resolve().parent
HEPDATA = HERE / "Input/atlas_hepdata_table14_v2.json"
ATLAS_PAPER = "https://arxiv.org/pdf/2211.11470"
PAPER_MODES = {"diagonal": (0.22, 0.18, 7.18),
               "covariant_log": (0.14, 0.25, 8.19)}
SEEDS = {"diagonal": (20261050, 20261052),
         "covariant_log": (20261051, 20261053)}
COLORS = {"diagonal": "tab:blue", "covariant_log": "tab:red"}
SOURCE_COLORS = {
    "JER": (255, 0, 0), "JES baseline": (0, 0, 255),
    "JES flavor": (0, 153, 0), "JES quenching": (153, 0, 153),
    "Unfolding": (51, 204, 204),
}


def spectrum(p, parameters):
    a, b, c = parameters
    return (a / p) ** (b + c * np.log(p)) / (1 + np.exp((p - 700) / 150))


def atlas_data():
    rows = json.loads(HEPDATA.read_text())["values"]
    centers = np.array([float(row["x"][0]["value"]) for row in rows])
    edges = [158.0]
    for center in centers:
        edges.append(2 * center - edges[-1])
    edges = np.array(edges)
    data = np.array([float(row["y"][0]["value"]) for row in rows])
    stat = np.array([float(row["y"][0]["errors"][0]["symerror"]) for row in rows])
    syst = np.array([float(row["y"][0]["errors"][1]["symerror"]) for row in rows])
    if len(data) != 10 or not np.isclose(edges[-1], 631):
        raise ValueError("unexpected ATLAS central table")
    return centers, edges, data, stat, syst


def digitize_covariance(paper, output, centers, data, stat, syst):
    """Read the five published relative-source curves in ATLAS Fig. 4."""
    image_path = output / "atlas_figure4_page13.png"
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m",
                    "-r250", "-dFirstPage=13", "-dLastPage=13",
                    f"-sOutputFile={image_path}", str(paper)], check=True)
    pixels = np.asarray(Image.open(image_path).convert("RGB"))
    relative = {}
    for name, rgb in SOURCE_COLORS.items():
        values = []
        for p in centers:
            x = round(1460.75 + 317.5 * np.log(p / 200.0))
            patch = pixels[350:820, x - 15:x + 16]
            y, _ = np.where(np.all(patch == rgb, axis=2))
            if len(y) < 4:
                raise ValueError(f"cannot digitize {name} at pT={p}")
            values.append(max(0.0, (821 - np.median(y + 350)) / 3125.0))
        relative[name] = np.array(values)
    absolute = {name: curve * data for name, curve in relative.items()}
    covariance = np.diag(stat**2) + sum(np.outer(v, v) for v in absolute.values())
    np.linalg.cholesky(covariance)
    total_digitized = np.sqrt(sum(v**2 for v in relative.values()))
    mismatch = np.max(np.abs(total_digitized - syst / data) / (syst / data))
    record = {
        "source": "ATLAS arXiv:2211.11470, Figure 4 top-right, page 13",
        "relative_sources": {name: values.tolist() for name, values in relative.items()},
        "absolute_covariance": covariance.tolist(),
        "max_fractional_total_systematic_mismatch": float(mismatch),
        "note": "unrescaled plot digitization; source covariance is approximate",
    }
    (output / "digitized_covariance.json").write_text(json.dumps(record, indent=2) + "\n")
    return covariance, mismatch


class ForwardModel:
    def __init__(self, edges, pp_parameters, step=0.125):
        # Integrate over each initial momentum. The gamma CDF difference is
        # the probability that its final momentum lands in an observed bin.
        self.edges = edges
        self.p = edges[0] + (np.arange(round((800 - edges[0]) / step)) + 0.5) * step
        self.weights = spectrum(self.p, pp_parameters) * step
        self.denominator = np.histogram(self.p, edges, weights=self.weights)[0]
        self.upper = np.maximum(self.p[:, None] - edges[:-1], 0)
        self.lower = np.maximum(self.p[:, None] - edges[1:], 0)

    def predict(self, a, b, k):
        if a == 0:
            return np.ones(len(self.denominator))
        mean = a * self.p**b * np.log(self.p)
        scale = mean / k
        probability = gammainc(k, self.upper / scale[:, None]) - \
                      gammainc(k, self.lower / scale[:, None])
        return (self.weights @ np.maximum(probability, 0)) / self.denominator


def design(count, seed):
    unit = qmc.LatinHypercube(d=3, seed=seed).random(count)
    return np.column_stack((10 ** (-2 + 3 * unit[:, 0]),
                            -1 + 2 * unit[:, 1], 1 + 9 * unit[:, 2]))


def features(parameters):
    return np.column_stack(((np.log10(parameters[:, 0]) + 2) / 3,
                            (parameters[:, 1] + 1) / 2,
                            (parameters[:, 2] - 1) / 9))


def train_emulator(forward, output, count):
    # The first 500 points and second 500 points are independent Latin
    # hypercubes. The fresh 100-point holdout is never used in fitting.
    first = design(count // 2, 20261030)
    second = design(count - count // 2, 20261032)
    training = np.vstack((first, second))
    heldout = design(max(10, count // 10), 20261033)
    response = np.vstack([forward.predict(*p) for p in training])
    truth = np.vstack([forward.predict(*p) for p in heldout])
    pca = PCA(n_components=4, svd_solver="full")
    scores = pca.fit_transform(response)
    gps = []
    for component in range(4):
        kernel = ConstantKernel(1, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(3), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=2, random_state=20261040 + component)
        gp.fit(features(training), scores[:, component])
        gps.append(gp)
    prediction = pca.inverse_transform(np.column_stack(
        [gp.predict(features(heldout)) for gp in gps]))
    error = prediction - truth
    report = {"training_points": len(training), "heldout_points": len(heldout),
              "heldout_rms_RAA": float(np.sqrt(np.mean(error**2))),
              "heldout_max_abs_RAA": float(np.max(np.abs(error)))}
    np.savez_compressed(output / "design.npz", training=training,
                        response=response, heldout=heldout, heldout_response=truth)
    joblib.dump({"pca": pca, "gps": gps}, output / "emulator.joblib")
    (output / "emulator_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return training, pca, gps, report


class Posterior:
    def __init__(self, kind, pca, gps, data, stat, syst, full_covariance):
        self.kind, self.pca, self.gps, self.data = kind, pca, gps, data
        diagonal = np.diag(stat**2 + syst**2)
        self.experimental = diagonal if kind == "diagonal" else \
            full_covariance / np.outer(data, data)

    def predict(self, theta):
        parameters = np.column_stack((np.exp(theta[:, 0]), theta[:, 1], theta[:, 2]))
        x = features(parameters)
        means, variances = [], []
        for gp in self.gps:
            mean, std = gp.predict(x, return_std=True)
            means.append(mean)
            variances.append(std**2)
        mean = self.pca.inverse_transform(np.column_stack(means))
        # Independent GP scores induce a full covariance across RAA bins.
        covariance = np.einsum("nc,ci,cj->nij", np.column_stack(variances),
                               self.pca.components_, self.pca.components_)
        return mean, covariance

    def log_probability(self, theta):
        theta = np.atleast_2d(theta)
        answer = np.full(len(theta), -np.inf)
        q, b, k = theta.T
        valid = (np.log(0.01) < q) & (q < np.log(10)) & \
                (-1 < b) & (b < 1) & (1 < k) & (k < 10)
        if not np.any(valid):
            return answer
        selected = theta[valid]
        mean, gp_covariance = self.predict(selected)
        positive = np.all(np.isfinite(mean), axis=1)
        if self.kind == "covariant_log":
            positive &= np.all(mean > 0, axis=1)
        mean, gp_covariance = mean[positive], gp_covariance[positive]
        if not len(mean):
            return answer
        if self.kind == "diagonal":
            residual = mean - self.data
            covariance = self.experimental + gp_covariance
        else:
            residual = np.log(mean) - np.log(self.data)
            covariance = self.experimental + gp_covariance / (mean[:, :, None] * mean[:, None, :])
        sign, logdet = np.linalg.slogdet(covariance)
        stable = sign > 0
        chi2 = np.einsum("ni,ni->n", residual[stable],
                         np.linalg.solve(covariance[stable], residual[stable, :, None])[:, :, 0])
        # Sampling coordinate q=ln(a) requires +q for a flat prior in a.
        lp = -0.5 * (chi2 + logdet[stable] + len(self.data) * np.log(2 * np.pi))
        lp += selected[positive][stable, 0]
        answer[np.flatnonzero(valid)[positive][stable]] = lp
        return answer


def run_chain(posterior, training, seed, walkers, steps, output):
    rng = np.random.default_rng(seed)
    candidates = np.column_stack((np.log(training[:, 0]), training[:, 1], training[:, 2]))
    top = np.argsort(posterior.log_probability(candidates))[-12::2]
    optimized = []
    for index in top:
        def objective(x):
            value = posterior.log_probability(x)[0]
            return -value if np.isfinite(value) else 1e100
        result = minimize(objective, candidates[index],
                          method="L-BFGS-B", bounds=((np.log(0.010001), np.log(9.999)),
                          (-0.999, 0.999), (1.001, 9.999)))
        if np.isfinite(result.fun):
            optimized.append((result.fun, result.x))
    center = min(optimized, key=lambda item: item[0])[1]
    initial = np.tile(center, (walkers, 1))
    initial[:, 0] = np.clip(center[0] + rng.normal(0, 0.14, walkers),
                            np.log(0.0101), np.log(9.9))
    initial[:, 1] = np.clip(center[1] + rng.normal(0, 0.055, walkers), -0.99, 0.99)
    initial[:, 2] = rng.uniform(1.05, 9.95, walkers)
    if not np.all(np.isfinite(posterior.log_probability(initial))):
        raise ValueError("invalid initial walkers")
    backend = emcee.backends.HDFBackend(output / f"{posterior.kind}_{seed}.h5")
    backend.reset(walkers, 3)
    sampler = emcee.EnsembleSampler(walkers, 3, posterior.log_probability,
                                    vectorize=True, backend=backend)
    started = time.perf_counter()
    sampler.run_mcmc(initial, steps, progress=False)
    burn = 500 if steps > 1000 else steps // 4
    sampled_q = backend.get_chain(discard=burn, flat=True)
    log_density_a = backend.get_log_prob(discard=burn, flat=True) - sampled_q[:, 0]
    physical = np.column_stack((np.exp(sampled_q[:, 0]), sampled_q[:, 1:]))
    report = {"seed": seed, "steps": steps,
              "mean_acceptance": float(np.mean(sampler.acceptance_fraction)),
              "median": np.median(physical, axis=0).tolist(),
              "physical_map": physical[np.argmax(log_density_a)].tolist(),
              "seconds": time.perf_counter() - started}
    (output / f"{posterior.kind}_{seed}.json").write_text(json.dumps(report, indent=2) + "\n")
    return physical, log_density_a, report


def contour(ax, x, y, xrange, yrange, color):
    histogram, xe, ye = np.histogram2d(x, y, bins=65, range=(xrange, yrange))
    histogram = gaussian_filter(histogram, 1)
    ranked = np.sort(histogram.ravel())[::-1]
    mass = np.cumsum(ranked) / ranked.sum()
    levels = sorted(ranked[np.searchsorted(mass, [0.95, 0.68])])
    ax.contour((xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2,
               histogram.T, levels=levels, colors=color)


def plot_posterior(samples, modes, forward, data, stat, syst, output):
    # Paper Table II reports a gamma scale coefficient. For approximate
    # comparison, our mean coefficient a maps to the paper's a*k.
    fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.5), layout="constrained")
    ranges = ((0, 7), (-0.25, 0.45), (1, 10))
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            if row == col:
                for kind in SEEDS:
                    values = samples[kind]
                    transformed = np.column_stack((values[:, 0] / values[:, 2],
                                                   values[:, 1], values[:, 2]))
                    ax.hist(transformed[:, row], bins=60, range=ranges[row],
                            histtype="step", density=True, color=COLORS[kind], label=kind)
                if row == 0:
                    ax.legend(fontsize=7)
            else:
                kind = "diagonal" if row > col else "covariant_log"
                values = samples[kind]
                transformed = np.column_stack((values[:, 0] / values[:, 2],
                                               values[:, 1], values[:, 2]))
                contour(ax, transformed[:, col], transformed[:, row],
                        ranges[col], ranges[row], COLORS[kind])
                paper = PAPER_MODES[kind]
                ax.plot(paper[col], paper[row], "k*", markersize=8)
                own = modes[kind]
                own_scale = np.array((own[0] / own[2], own[1], own[2]))
                ax.plot(own_scale[col], own_scale[row], marker="x",
                        color=COLORS[kind], markersize=7)
            ax.set_xlim(ranges[col])
            if row != col:
                ax.set_ylim(ranges[row])
            if row == 2:
                ax.set_xlabel(("effective scale a/k", "b", "k")[col])
            if col == 0 and row > 0:
                ax.set_ylabel(("effective scale a/k", "b", "k")[row])
    fig.savefig(output / "figure4_posterior.pdf")
    plt.close(fig)

    centers = (forward.edges[:-1] + forward.edges[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout="constrained")
    for ax, kind in zip(axes, SEEDS):
        ax.errorbar(centers, data, yerr=np.hypot(stat, syst), fmt="ko", label="ATLAS")
        ax.plot(centers, forward.predict(*modes[kind]), color=COLORS[kind], label="physical MAP")
        ax.set(xlabel=r"jet $p_T$ [GeV]", title=kind.replace("_", " "))
        ax.legend()
    axes[0].set_ylabel(r"$R_{AA}$")
    fig.savefig(output / "raa_at_map.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-paper", type=Path,
                        help="local ATLAS PDF; otherwise download arXiv:2211.11470")
    parser.add_argument("--training-points", type=int, default=1000)
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "Output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paper = args.atlas_paper or args.output / "ATLAS_2211.11470.pdf"
    if not paper.exists():
        response = requests.get(ATLAS_PAPER, timeout=90)
        response.raise_for_status()
        paper.write_bytes(response.content)
    centers, edges, data, stat, syst = atlas_data()
    covariance, mismatch = digitize_covariance(paper, args.output,
                                                 centers, data, stat, syst)
    pp = json.loads((HERE / "Input/pp_fit.json").read_text())["parameters"]
    forward = ForwardModel(edges, pp)
    if args.check_only:
        print(json.dumps({"zero_loss_max_abs_RAA_minus_one":
                          float(np.max(np.abs(forward.predict(0, 0, 2) - 1))),
                          "covariance_total_mismatch_fraction": float(mismatch)}, indent=2))
        return
    training, pca, gps, validation = train_emulator(forward, args.output,
                                                    args.training_points)
    samples, modes, reports = {}, {}, {}
    for kind, seeds in SEEDS.items():
        posterior = Posterior(kind, pca, gps, data, stat, syst, covariance)
        chains = [run_chain(posterior, training, seed, args.walkers,
                            args.steps, args.output) for seed in seeds]
        samples[kind] = np.vstack([chain[0] for chain in chains])
        density = np.concatenate([chain[1] for chain in chains])
        modes[kind] = samples[kind][np.argmax(density)]
        reports[kind] = [chain[2] for chain in chains]
    plot_posterior(samples, modes, forward, data, stat, syst, args.output)
    summary = {"prior": "flat physical a 0.01-10, b -1-1, k 1-10",
               "emulator_error": "location-dependent four-PC GP covariance only",
               "training_validation": validation,
               "source_covariance_mismatch_fraction": float(mismatch),
               "physical_map": {kind: modes[kind].tolist() for kind in SEEDS},
               "chains": reports}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
