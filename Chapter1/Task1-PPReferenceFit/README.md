# Task 1: pp reference fit

Run `python Task1-PPReferenceFit.py` in this directory. It creates
`Output/pp_fit.json` and `Output/pp_fit.pdf`.

Input: `Input/hepdata_ins1673184_table4.json`, the unchanged HEPData
[Table 4](https://www.hepdata.net/download/table/ins1673184/Table%204/json)
for the ATLAS 5.02 TeV pp jet spectrum. This task uses nine bins from
100 to 800 GeV. The source table labels `|y|<2.8`; the ezQuench paper
describes `|y|<2.1`, so preserve that source caveat.

The model is `(A/p)^(B+C ln p)/(1+exp((p-700)/150))`. It is compared to
1 GeV sampled bin averages. A single Gaussian nuisance parameter shifts
all bins along their average systematic errors. The JSON's `parameters`
array is the fixed pp baseline for Tasks 3, 4, and 5. Copy
`Output/pp_fit.json` into each task's `Input/pp_fit.json` before running it.

Requirements: Python with NumPy, SciPy, and Matplotlib. The accepted
baseline is approximately `(103.79175, 2.50053, 0.59143)`.
