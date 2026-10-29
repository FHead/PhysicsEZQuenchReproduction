# Task 4: central-only delta-loss fit

Copy Task 1's `Output/pp_fit.json` to this task's `Input/pp_fit.json`, then
run `python Task4-CentralOnlyDeltaLossFit.py`. The script directly evaluates
the deterministic energy-loss model for two independent `emcee` chains per
likelihood and writes its chains, summaries, and Figure 5-style posterior
plot into ignored `Output/`. `--check-only` performs the zero-loss check
without running MCMC.

The loss is `a p^b ln(p)` with a flat prior `0<a<10`, `-1<b<10`.
The pp spectrum comes from Task 1's copied output. The ten central 0–10%
ATLAS `R_AA` points and covariance matrices come from the unchanged public
`Input/ATLAS_data_public.h5` source (304 KiB), available as
[ATLAS_data.h5 in the ez-Quench repository](https://github.com/LLNL/ez-Quench/blob/main/h5/ATLAS_data.h5).
This is **data provenance**, not
code reuse: the file contains covariance components not published in
machine-readable ATLAS HEPData. The diagonal fit uses linear `R_AA` with
`C_stat + diag(C_sys,total)`. The covariant fit uses log `R_AA` with
`(C_stat + C_sys,sum)/(R_i R_j)`.

The script uses a 0.03 GeV initial-momentum grid, 24 walkers, and 4000/5000
steps for first/repeat chains. Its posterior plot marks the paper Table II
modes `(1.41,0.20)` and `(1.08,0.26)` for comparison. It contains no TRENTO
input. Requirements: NumPy, SciPy, h5py, emcee, and Matplotlib.
