"""Reproduce four-centrality TRENTO delta-loss RAA and its posterior."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import warnings

_CACHE = Path(__file__).resolve().parent / "Output/cache"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import emcee
import h5py
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


HERE = Path(__file__).resolve().parent
DEFAULT_PROFILES = HERE / "Input/PbPb_50000_seed12345.h5"
DEFAULT_INDEX = HERE / "Input/PbPb_50000_seed12345.index.npz"
EDGES = np.array([158., 178., 200., 224., 251., 282., 316.,
                  355., 398., 501., 631.])


def spectrum(p, parameters):
    a, b, c = parameters
    return (a / p) ** (b + c * np.log(p)) / (1 + np.exp((p - 700) / 150))


def ray_integral(rho, x0, y0, angle, tau=0.5):
    """Integrate retained ray cells with first-retained-midpoint/r dilution."""
    n = rho.shape[0]
    width = 20.0 / n
    dx, dy = np.cos(angle), np.sin(angle)
    tx = ((10 if dx > 0 else -10) - x0) / dx if abs(dx) > 1e-14 else np.inf
    ty = ((10 if dy > 0 else -10) - y0) / dy if abs(dy) > 1e-14 else np.inf
    exit_distance = min(tx, ty)
    if exit_distance <= 0:
        return 0.0
    edges = np.linspace(-10, 10, n + 1)
    crossings = [np.array([0, exit_distance])]
    if abs(dx) > 1e-14:
        t = (edges - x0) / dx
        crossings.append(t[(t > 0) & (t < exit_distance)])
    if abs(dy) > 1e-14:
        t = (edges - y0) / dy
        crossings.append(t[(t > 0) & (t < exit_distance)])
    t = np.sort(np.concatenate(crossings))
    length = np.diff(t)
    valid = length > 1e-12
    midpoint = ((t[1:] + t[:-1]) / 2)[valid]
    length = length[valid]
    ix = np.clip(np.floor((x0 + midpoint * dx + 10) / width).astype(int), 0, n - 1)
    iy = np.clip(np.floor((y0 + midpoint * dy + 10) / width).astype(int), 0, n - 1)
    retained = midpoint > tau
    if not np.any(retained):
        return 0.0
    onset = midpoint[retained][0]
    return float(np.sum(length[retained] * rho[iy[retained], ix[retained]] *
                        onset / midpoint[retained]))


def sample_paths(profiles, index, centrality, pairs, rng):
    """Draw event by W, cell by rho²/W, then full-azimuth back-to-back rays."""
    eligible = np.flatnonzero(index["class_label"] == centrality)
    choices = rng.choice(eligible, size=pairs,
                         p=index["event_probability_within_class"][eligible])
    angles = rng.uniform(0, 2 * np.pi, pairs)
    result = np.empty((pairs, 2))
    for i, event_id in enumerate(choices):
        rho = profiles[index["event_names"][event_id]][...]
        rho2 = np.square(rho).ravel()
        # A cumulative draw avoids normalizing and scanning 10,000 cells
        # inside NumPy's general categorical sampler for every jet.
        cell = int(np.searchsorted(np.cumsum(rho2), rng.random()*rho2.sum()))
        iy, ix = divmod(cell, rho.shape[1])
        x = -10 + 20 / rho.shape[1] * (ix + rng.random())
        y = -10 + 20 / rho.shape[0] * (iy + rng.random())
        result[i] = (ray_integral(rho, x, y, angles[i]),
                     ray_integral(rho, x, y, angles[i] + np.pi))
    return result


def raa(momentum, weights, paths, a, b):
    denominator = np.histogram(momentum, EDGES, weights=weights)[0]
    if a == 0:
        return np.ones(10)
    loss = a * momentum**b * np.log(momentum)
    numerator = np.zeros(10)
    for pair in paths:
        for path in pair:
            numerator += np.histogram(momentum - path * loss, EDGES,
                                      weights=weights)[0]
    return numerator / (2 * len(paths) * denominator)


def designs():
    """Recreate the accepted 73+200+20 design, independent of data fits."""
    broad = qmc.LatinHypercube(d=2, scramble=True, seed=20261030).random(400)
    broad = qmc.scale(broad, [0, -1], [10, 10])
    seeds = np.random.SeedSequence(20261031).generate_state(400)
    points = [(a, b, int(seeds[i]), 0) for i, (a, b) in enumerate(broad) if b < 1]

    supplemental = qmc.LatinHypercube(d=2, scramble=True, seed=20261101).random(200)
    loga = -2 + 3 * supplemental[:, 0]
    b = -1 + 2 * supplemental[:, 1]
    seeds = np.random.SeedSequence(20261102).generate_state(200)
    points += [(10**loga[i], b[i], int(seeds[i]), 1) for i in range(200)]

    patch = qmc.LatinHypercube(d=2, scramble=True, seed=20261104).random(20)
    loga = np.log10(0.03) + patch[:, 0]
    b = 0.5 + 0.5 * patch[:, 1]
    seeds = np.random.SeedSequence(20261105).generate_state(20)
    points += [(10**loga[i], b[i], int(seeds[i]), 2) for i in range(20)]
    if len(points) != 293:
        raise ValueError("wrong accepted design size")
    return [{"id": i, "a": float(a), "b": float(b), "seed": seed,
             "campaign": campaign} for i, (a, b, seed, campaign) in enumerate(points)]


def evaluate_point(point, profiles, index, pp, pair_count, output):
    rng = np.random.default_rng(point["seed"])
    momentum = np.arange(100.5, 800.0, 1.0)
    weights = spectrum(momentum, pp)
    prediction = np.empty((4, 10))
    with np.load(index) as archive, h5py.File(profiles, "r") as events:
        data = {key: archive[key] for key in ("class_label", "event_names",
                "event_probability_within_class")}
        for centrality in range(4):
            paths = sample_paths(events, data, centrality, pair_count, rng)
            prediction[centrality] = raa(momentum, weights, paths,
                                         point["a"], point["b"])
    path = output / "points" / f"{point['id']:03d}.npz"
    np.savez_compressed(path, raa=prediction, a=point["a"], b=point["b"],
                        seed=point["seed"], pairs_per_class=pair_count)
    return point["id"]


def run_forward(points, profiles, index, pp, pairs, workers, output, limit):
    folder = output / "points"
    folder.mkdir(parents=True, exist_ok=True)
    for point in points:
        path = folder / f"{point['id']:03d}.npz"
        if path.exists():
            with np.load(path) as saved:
                if (int(saved["pairs_per_class"]) != pairs or
                    int(saved["seed"]) != point["seed"] or
                    float(saved["a"]) != point["a"] or
                    float(saved["b"]) != point["b"] or
                    saved["raa"].shape != (4, 10)):
                    raise ValueError(f"stored forward point has different settings: {path}")
    pending = [point for point in points if not (folder / f"{point['id']:03d}.npz").exists()]
    if limit is not None:
        pending = pending[:limit]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(evaluate_point, p, profiles, index, pp, pairs, output)
                for p in pending]
        for job in as_completed(jobs):
            print(f"forward {job.result() + 1}/{len(points)}", flush=True)
    return len(list(folder.glob("*.npz")))


def training_data(points, output):
    """Load complete forward calculations in design order."""
    files = [output / "points" / f"{point['id']:03d}.npz" for point in points]
    if not all(path.exists() for path in files):
        raise ValueError("forward design is incomplete")
    y = np.stack([np.load(path)["raa"].reshape(40) for path in files])
    x = np.array([[(np.log10(p["a"])+2)/3, (p["b"]+1)/2]
                  for p in points])
    if not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("invalid forward response")
    return x, y


def fit_gp(x, y):
    """Fit ten PCA scores with independent Matérn Gaussian processes."""
    pca = PCA(n_components=10, svd_solver="full")
    scores = pca.fit_transform(y)
    models = []
    for column in scores.T:
        kernel = (ConstantKernel(1, (1e-3, 1e3)) *
                  Matern(length_scale=[0.2, 0.2],
                         length_scale_bounds=(1e-3, 10), nu=2.5) +
                  WhiteKernel(noise_level=1e-5,
                              noise_level_bounds=(1e-9, 1e-1)))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8,
                                      normalize_y=True, n_restarts_optimizer=0,
                                      random_state=20261103)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(x, column)
        models.append(gp)
    return pca, models


def predict_scores(models, x):
    means, stds = zip(*(model.predict(x, return_std=True) for model in models))
    return np.stack(means, axis=1), np.stack(stds, axis=1)


def loo_fold(i, x, y, folder):
    mask = np.arange(len(x)) != i
    pca, models = fit_gp(x[mask], y[mask])
    score, std = predict_scores(models, x[[i]])
    np.savez_compressed(folder / f"{i:03d}.npz",
                        prediction=pca.inverse_transform(score)[0],
                        oracle=pca.inverse_transform(pca.transform(y[[i]]))[0],
                        score_std=std[0],
                        score_error=pca.transform(y[[i]])[0]-score[0],
                        components=pca.components_)
    return i


def validate_loo(points, x, y, workers, output):
    folder = output / "loo"
    folder.mkdir(exist_ok=True)
    pending = [i for i in range(len(points)) if not (folder / f"{i:03d}.npz").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(loo_fold, i, x, y, folder) for i in pending]
        for job in as_completed(jobs):
            print(f"LOO {job.result()+1}/{len(points)}", flush=True)
    folds = [np.load(folder / f"{i:03d}.npz") for i in range(len(points))]
    prediction = np.stack([f["prediction"] for f in folds])
    oracle = np.stack([f["oracle"] for f in folds])
    std = np.stack([f["score_std"] for f in folds])
    error = np.stack([f["score_error"] for f in folds])
    basis = np.stack([f["components"] for f in folds])
    campaigns = np.array([p["campaign"] for p in points])
    rng = np.random.default_rng(20261106)
    check = np.sort(np.concatenate([rng.choice(np.flatnonzero(campaigns == c), n,
                                               replace=False)
                                    for c, n in enumerate((15, 40, 5))]))
    calibration = np.setdiff1d(np.arange(len(points)), check)
    raw_scale = np.sqrt(np.mean((error[calibration]/std[calibration])**2, axis=0))
    trunc = y[calibration]-oracle[calibration]
    trunc_cov = trunc.T @ trunc / len(calibration)

    def coverage(ids, factor):
        variance = np.einsum("ij,ijk->ik",
                             (std[ids]*raw_scale*factor)**2, basis[ids]**2)
        variance += np.diag(trunc_cov)
        return np.mean(np.abs(prediction[ids]-y[ids]) <= 2*np.sqrt(variance))

    def adequate(factor):
        return all(coverage(calibration[campaigns[calibration] == c], factor)
                   >= 0.95 for c in range(3))

    high = 1.0
    while not adequate(high):
        high *= 2
    low = 1.0
    for _ in range(40):
        middle = (low+high)/2
        if adequate(middle):
            high = middle
        else:
            low = middle
    scale = raw_scale*high
    np.savez_compressed(output / "validation.npz", true=y, prediction=prediction,
                        oracle=oracle, score_std=std, score_error=error,
                        components=basis, calibration_ids=calibration,
                        check_ids=check, score_scale=scale, trunc_cov=trunc_cov)
    report = {"loo_rms": float(np.sqrt(np.mean((prediction-y)**2))),
              "loo_max_abs": float(np.max(np.abs(prediction-y))),
              "check_2sigma_coverage": float(coverage(check, high)),
              "score_scale": scale.tolist(), "safety_factor": high}
    (output / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def experimental_data(path):
    with h5py.File(path, "r") as source:
        observed = source["RAA_vals"][...].reshape(40)
        covariance = (source["cov_RAA_stat_diag"][...] +
                      np.diag(np.diag(source["cov_RAA_sys_total"][...])) +
                      source["cov_pp_lum_E"][...] + source["cov_TAA_E"][...])
    np.linalg.cholesky(covariance)
    return observed, covariance


class Posterior:
    def __init__(self, pca, models, scale, trunc_cov, observed, covariance):
        self.pca, self.models, self.scale = pca, models, scale
        self.trunc_cov, self.observed, self.covariance = trunc_cov, observed, covariance

    def predict(self, theta):
        a, b = theta
        x = np.array([[(np.log10(a)+2)/3, (b+1)/2]])
        score, std = predict_scores(self.models, x)
        components = self.pca.components_
        emulator = ((components.T*(std[0]*self.scale)**2) @ components
                    + self.trunc_cov)
        return self.pca.inverse_transform(score)[0], emulator

    def log_probability(self, theta):
        a, b = theta
        if not (0.01 < a < 10 and -1 < b < 1):
            return -np.inf
        prediction, emulator = self.predict(theta)
        factor = np.linalg.cholesky(self.covariance+emulator)
        white = solve_triangular(factor, prediction-self.observed,
                                 lower=True, check_finite=False)
        return float(-0.5*(white@white+2*np.log(np.diag(factor)).sum()
                            +len(white)*np.log(2*np.pi)))


def run_chain(posterior, seed, steps, walkers, output):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    path = output / f"chain_seed{seed}.h5"
    fresh = not path.exists()
    backend = emcee.backends.HDFBackend(path)
    if fresh or backend.iteration == 0:
        backend.reset(walkers, 2)
        optimum = differential_evolution(
            lambda x: -posterior.log_probability((10**x[0], x[1])),
            [(-1.999, 0.999), (-0.999, 0.999)], seed=seed,
            maxiter=80, popsize=12, polish=True)
        center = (10**optimum.x[0], optimum.x[1])
        initial = np.column_stack((
            np.clip(10**rng.normal(np.log10(center[0]), 0.2, walkers),
                    0.010001, 9.999),
            np.clip(rng.normal(center[1], 0.1, walkers), -0.999, 0.999)))
    else:
        initial = None
    sampler = emcee.EnsembleSampler(
        walkers, 2, posterior.log_probability, backend=backend,
        moves=[(emcee.moves.StretchMove(), 0.8), (emcee.moves.DEMove(), 0.2)])
    while backend.iteration < steps:
        sampler.run_mcmc(initial, min(500, steps-backend.iteration), progress=False)
        initial = None
        print(f"chain {seed}: {backend.iteration}/{steps}", flush=True)
    return backend.get_chain(discard=min(steps//4, 1000), flat=True)


def plots(points, y, posterior, samples, output):
    coordinates = np.array([[p["a"], p["b"]] for p in points])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(coordinates[:, 0], coordinates[:, 1], c=np.max(y, axis=1),
                    s=16, cmap="viridis")
    axes[0].set(xscale="log", xlabel="a", ylabel="b", title="293 design points")
    axes[1].hist2d(samples[:, 0], samples[:, 1], bins=60, cmap="viridis")
    axes[1].set(xlabel="a", ylabel="b", title="Four-centrality posterior")
    fig.tight_layout()
    fig.savefig(output / "design_and_posterior.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    centers = (EDGES[:-1]+EDGES[1:])/2
    for c, ax in enumerate(axes.flat):
        ax.errorbar(centers, posterior.observed.reshape(4, 10)[c],
                    yerr=np.sqrt(np.diag(posterior.covariance).reshape(4, 10)[c]),
                    fmt="o", color="black")
        for theta in samples[::max(1, len(samples)//100)][:100]:
            ax.plot(centers, posterior.predict(theta)[0].reshape(4, 10)[c],
                    color="tab:blue", alpha=0.04)
        ax.set(title=("0–10%", "10–30%", "30–50%", "50–80%")[c],
               ylabel=r"$R_{AA}$", ylim=(0, 1.1))
    axes[-1, 0].set_xlabel("Jet $p_T$ (GeV)")
    axes[-1, 1].set_xlabel("Jet $p_T$ (GeV)")
    fig.tight_layout()
    fig.savefig(output / "raa_posterior.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--pairs", type=int, default=25000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--stage", choices=("all", "forward", "loo", "fit", "posterior"),
                        default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "Output")
    args = parser.parse_args()
    points = designs()
    if args.check_only:
        uniform = np.ones((1, 1))
        opposite = [ray_integral(uniform, 0, 0, angle)
                    for angle in (0, np.pi)]
        if not np.allclose(opposite, 10):
            raise ValueError("single-cell path integral failed")
        p = np.arange(100.5, 800, 1)
        if not np.array_equal(raa(p, np.ones(len(p)), np.ones((2, 2)), 0, 0),
                              np.ones(10)):
            raise ValueError("zero-loss RAA is not one")
        observed, covariance = experimental_data(HERE / "Input/ATLAS_data_public.h5")
        print(json.dumps({"design_count": len(points),
                          "campaign_counts": [sum(p["campaign"] == c for p in points)
                                              for c in range(3)],
                          "uniform_path_integral": opposite[0],
                          "observed_shape": list(observed.shape),
                          "covariance_shape": list(covariance.shape)}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    pp = json.loads((HERE / "Input/pp_fit.json").read_text())["parameters"]
    if args.stage in ("all", "forward"):
        count = run_forward(points, args.profiles, args.index, pp,
                            args.pairs, args.workers, args.output, args.limit)
        print(f"forward complete: {count}/{len(points)}")
        if args.stage == "forward":
            return
    x, y = training_data(points, args.output)
    if args.stage in ("all", "loo") and not (args.output / "validation.npz").exists():
        print(json.dumps(validate_loo(points, x, y, args.workers, args.output), indent=2))
    if args.stage == "loo":
        return
    model_path = args.output / "emulator.joblib"
    if args.stage in ("all", "fit") or not model_path.exists():
        pca, models = fit_gp(x, y)
        joblib.dump((pca, models), model_path)
    else:
        pca, models = joblib.load(model_path)
    if args.stage == "fit":
        return
    with np.load(args.output / "validation.npz") as validation:
        scale = validation["score_scale"].copy()
        trunc_cov = validation["trunc_cov"].copy()
    observed, covariance = experimental_data(HERE / "Input/ATLAS_data_public.h5")
    posterior = Posterior(pca, models, scale, trunc_cov, observed, covariance)
    chains = [run_chain(posterior, seed, args.steps, args.walkers, args.output)
              for seed in (20261108, 20261109)]
    samples = np.concatenate(chains)
    report = {"prior": "flat 0.01<a<10, -1<b<1", "pairs_per_class": args.pairs,
              "design_points": len(points), "steps_per_chain": args.steps,
              "walkers": args.walkers, "posterior_median": np.median(samples, axis=0).tolist(),
              "posterior_16_84": np.quantile(samples, [0.16, 0.84], axis=0).tolist(),
              "emulator_error": "location-dependent calibrated GP score covariance plus PCA truncation"}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    plots(points, y, posterior, samples, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
