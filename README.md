# topo-ert-ip-tests

Synthetic tests of topographic effects in 2.5-D electrical resistivity tomography (ERT) and
induced polarisation (IP) for a dipole-dipole profile (a = 20 m, n = 1-5, 46 electrodes over
900 m, about 72 m of topographic relief).

The code
- computes the pure topographic effect on apparent resistivity (homogeneous earth under the real
  topography of the test profile),
- validates the 2.5-D finite-element forward solver against the analytic half-space and checks
  mesh convergence,
- generates synthetic data for three resistivity models (A, B, C) with 3 % and 5 % noise and
  inverts them with a flat-earth and a topography-corrected mesh (L2 and robust L1, several
  vertical-to-horizontal smoothness weights),
- tests whether the chargeable and the conductive bodies are recovered at the right place
  (coincident and offset scenarios, Seigel linear IP inversion),
- optionally cross-checks the topographic effect and models A with pyGIMLi.

**The field data used in the paper are confidential and are not included.**
Only the survey *geometry* of the test profile is provided (`data/`): electrode positions,
four-electrode configurations and the topography. No measured values are needed to run the tests.

## Contents

| Path | Purpose |
|---|---|
| `data/geometry_electrodes.csv` | electrode index and position along the profile (`x_m`, 0-900 m) |
| `data/geometry_configurations.csv` | the 177 dipole-dipole configurations: electrode indices `C1, C2, P1, P2` and `n` |
| `data/geometry_topography.csv` | `x_m`, `elevation_m` of the 39 topography stations |
| `scripts/ert25d.py` | 2.5-D finite-element forward solver, Gauss-Newton inversion, Seigel IP inversion |
| `scripts/setup_geom.py` | reads the geometry, builds the meshes |
| `scripts/synth_common.py` | true models, metrics (log-misfit, IoU, centroid error), section plots |
| `scripts/t01`-`t16` | the individual steps (see `run_all.py`) |
| `scripts/journal_style.py`, `ert_plot_style.py` | figure style |
| `scripts/run_all.py` | runs all steps in order (resumable) |
| `scripts/check_reproducibility.py` | compares the computed metrics with the values reported in the paper |
| `pygimli/` | optional pyGIMLi cross-check of the synthetic tests |

## How to run

```bash
pip install -r requirements.txt
cd scripts
python run_all.py                 # all steps, roughly 30-40 min on one CPU core; logs in ../logs/
python check_reproducibility.py   # must print "REPRODUCIBILITY: ALL MATCH"
```

`run_all.py` is resumable: a step is skipped when its result file exists (delete it to redo the
step). Run single steps with e.g. `python run_all.py t03_topo_effect`.
Result files (`*.npy`, `*.npz`, `*.json`) and figures are written to `scripts/`.

Optional pyGIMLi cross-check (needs pyGIMLi, e.g. `conda create -n pg -c gimli -c conda-forge pygimli`):

```bash
cd pygimli
python check_pygimli_env.py
python pygimli_reproduce_synthetic.py
```

## Figures and tables of the paper

| Paper | Produced by | Output file in `scripts/` |
|---|---|---|
| Fig. 1 | (observed field data - not included) | - |
| Fig. 2 | `t10_plots.py` (numbers from `t01_validate.py`, `t02_meshconv.py`, `t04_topo_convergence.py`) | `F01_validation` |
| Fig. 3 | `t03_topo_effect.py` | `F02_topographic_effect` |
| Fig. 4 | `t06_invert.py` (model A), `t07_plot_A.py` | `F03_testA_sections` |
| Fig. 5 | `t13_geometryB.py`, `t15_plot_B.py` | `F08_geometryB` |
| Fig. 6 | `t16_geometryC.py` | `F09_geometryC` |
| Fig. 7 | `t06_invert.py` (5 % noise), `t10_plots.py` | `F04_noise5` |
| Fig. 8 | `t05_make_data.py`, `t10_plots.py` | `F06_pseudo_offset_coincident` |
| Fig. 9 | `t08_ip.py`, `t10_plots.py` | `F05_ip_offset_test` |
| Fig. 10 | (field data - not included) | - |
| Table 1 (synthetic metrics: chi2, log-misfit, IoU, centroid error) | `t06_invert.py`, `t13_geometryB.py`, `t16_geometryC.py` (`met_*.json`), IP offsets from `t08_ip.py` (`ipmet_*.json`); checked by `check_reproducibility.py` | logs in `../logs/` |

## Requirements

Python 3 with numpy, scipy, pandas and matplotlib (see `requirements.txt`).
pyGIMLi is needed only for the optional `pygimli/` scripts.

## Licence

MIT, see `LICENSE`.
