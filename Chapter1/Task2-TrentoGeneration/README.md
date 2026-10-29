# Task 2: TRENTO medium generation

Run `python Task2-TrentoGeneration.py --trento /path/to/trento` in this
directory. The default is 50,000 minimum-bias Pb+Pb events with seed 12345.
Use `--events 100` for a quick smoke run. The `Output/` directory will contain
the large HDF5 profile ensemble, a small `.index.npz`, and a JSON run record.
All three files are generated and ignored by Git. If the profile file already
exists, `--reuse-profiles` rebuilds its index without rerunning TRENTO.

TRENTO is the external [Duke-QCD/trento](https://github.com/Duke-QCD/trento)
program, version 2.0. The original run used commit
`7ef92367c51d2ef1a8946973d17a90fdd3c31f5e`.
Build and put its executable on `PATH`, or pass `--trento`. With Boost 1.92,
the source's CMake C++ standard must be raised from 11 to 14 before building.
The Python script records the complete TRENTO command in its JSON output.

Settings: reduced thickness p=0, gamma fluctuation 1, nucleon width 1 fm,
minimum distance 1.4 fm, cross section 6.4 fm², and a 20×20 fm transverse
grid at 0.2 fm spacing. The index ranks events by
`npart + mult/(max(mult)+1)` and assigns 0–10%, 10–30%, 30–50%, and 50–80%
classes. It also computes each event's relative hard-jet yield
`W = sum_cells rho²` and the event sampling probabilities within each class.

Task 3 needs copies of both `Output/PbPb_50000_seed12345.h5` and
`Output/PbPb_50000_seed12345.index.npz` in its `Input/` directory.
Requirements: Python with NumPy and h5py, plus TRENTO and its build libraries.
