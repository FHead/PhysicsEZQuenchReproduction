# Task 3: TRENTO delta-loss reproduction

Copy these **outputs**, not code, into `Input/`:

- Task 1 `Output/pp_fit.json` → `Input/pp_fit.json`.
- Task 2 `Output/PbPb_50000_seed12345.h5` and matching `.index.npz` → `Input/`.

The tracked `Input/ATLAS_data_public.h5` is the unchanged experimental-data
archive distributed with the public ez-Quench repository. Its covariance
matrices are an external analysis input and a caveat to this reproduction.

Run `python Task3-DeltaLossReproduction.py`. The script creates the accepted
293-point design (73 retained full-prior points, 200 log-amplitude points,
20 high-b patch points), evaluates 25,000 back-to-back dijet pairs per
centrality and design point, validates a 10-component PCA+GP with 293
leave-one-out folds, calibrates its location-dependent uncertainty, and runs
two 32-walker, 5,000-step posterior chains. Each forward point, validation
fold, and chain is checkpointed inside ignored `Output/`. `--stage` can run
`forward`, `loo`, `fit`, or `posterior` separately. `--limit N --stage forward`
is a smoke test; use a separate `--output` for pilots so they cannot mix with
production checkpoints. `--check-only` tests design counts, zero loss, a
single-cell ray integral, and experimental covariance. `--profiles` and `--index` can select
a pilot ensemble.

The production point is drawn from event weight `W=sum(rho²)`, then from
cell `rho²/W`. Its position is uniform within the cell, the azimuth is
uniform over a full turn, and the other jet points oppositely. Each ray
integrates density after 0.5 fm/c with first retained midpoint divided by
distance. Initial pp quadrature weights are shifted by
`a p_initial^b ln(p_initial)` times the path integral and histogrammed.
The GP input is `(log10(a)+2)/3, (b+1)/2`, and the prior is flat in physical
`a,b` over `0.01<a<10, -1<b<1`. The Gaussian likelihood uses linear
40-bin `R_AA`, the public semi-diagonal covariance, calibrated GP score
uncertainty, and PCA truncation covariance. Generated arrays, emulator,
chains, reports, and figures all stay in ignored `Output/`.
