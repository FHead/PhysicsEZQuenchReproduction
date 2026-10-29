# Physics ezQuench reproduction

Each task under `Chapter1/` is independently runnable. Run the similarly named
Python script in that task folder; its README lists the required external inputs
and any final output to copy from an earlier task. No task imports or executes
code from another task folder.

Generated files go in `Output/` and are ignored by Git. The task scripts recreate
them. Small, unchanged source data are kept in `Input/` with provenance in the
corresponding README. The published ezQuench code is used only as a reference for
understanding; no code from it is executed here.

The detailed methods and exploratory checks are recorded separately in the
technical note. This repository contains the minimal code for rerunning the
accepted Chapter 1 calculations.

Create a Python virtual environment and install the pinned Python packages:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run Tasks 1 and 2 first. Task 2 additionally needs the external TRENTO 2.0
executable (build instructions and exact physics settings are in its README).
Copy their specified outputs into the later tasks' `Input/` folders, then run
the similarly named script in each task folder. Task 4 and Task 5 can run
independently after Task 1. Task 3 needs both Task 1 and Task 2 outputs.
Each task README documents the exact copy names, model conventions, expected
figures, and command-line controls.

`Output/`, `Output-*/`, copied outputs in `Input/`, virtual environments,
chains, emulator files, and every other product of our scripts are ignored
by Git, regardless of size. The tracked `Input/` files are unchanged
external measurements or data archives.
