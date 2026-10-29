# Task 5: central gamma-loss reproduction

Copy Task 1's `Output/pp_fit.json` to `Input/pp_fit.json`, then run
`python Task5-GammaFitReproduction.py`. The script downloads the original
[ATLAS paper](https://arxiv.org/pdf/2211.11470) once, renders page 13 with
Ghostscript (`gs`), digitizes the five systematic-source curves in Figure 4, creates
the design and held-out calculations, trains a four-component PCA+GP,
validates it, runs two chains for each of the two likelihoods, and writes
posterior and `R_AA` plots. All generated files are under ignored `Output/`.
To work offline, pass `--atlas-paper /path/to/ATLAS_2211.11470.pdf`.
`--check-only` validates the source digitization and zero-loss limit.

The unchanged source `Input/atlas_hepdata_table14_v2.json` is
[HEPData Table 14](https://www.hepdata.net/download/table/ins2512925/Table%2014/json):
ten inclusive 0–10% ATLAS jet `R_AA` points. We use the paper-stated gamma
law, conditioned on **initial** jet momentum: mean loss
`mu(p)=a p^b ln(p)`, shape `k`, scale `mu/k`. We integrate gamma CDF bin
probabilities against the Task 1 pp spectrum. The paper's public helper
uses an approximately different coefficient convention and final-momentum
conditioning, so parameter comparison uses `our a/k ≈ paper a`.

The design is two independent 500-point Latin hypercubes (seeds 20261030
and 20261032) in log10(a)=[−2,1], b=[−1,1], k=[1,10], plus a fresh
100-point holdout (seed 20261033). Four independent Matérn-5/2 GPs emulate
PCA scores. The first posterior uses the validated restricted flat prior
0.01<a<10, −1<b<1, 1<k<10. The diagonal fit uses linear `R_AA` and HEPData
statistical-plus-total errors. The covariant fit uses log `R_AA` and an
approximate source covariance digitized from ATLAS Figure 4. Only the
location-dependent GP predictive covariance is added as emulator error.
Both likelihoods include the Gaussian log determinant. Two 32-walker,
4000-step `emcee` chains are run per likelihood.

Inputs: Task 1's copied pp JSON, the committed unchanged HEPData table,
and the downloaded or supplied ATLAS paper. Requirements: Python with
NumPy, SciPy, scikit-learn, emcee, joblib, Matplotlib, Pillow, requests,
and the Ghostscript `gs` command. Accepted reference checks: fresh
held-out RMS `R_AA` error about 0.00156, largest error about 0.00981;
physical MAPs around `(1.59,0.187,7.31)` and `(1.19,0.255,7.49)`.
The source digitization is approximate; the Ghostscript rendering used here
gives a maximum 5.3% mismatch between source quadrature and HEPData's
reported total systematic uncertainty.
