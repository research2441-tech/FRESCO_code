# FRESCO

Code and results for the manuscript *FRESCO: Freshness-Aware Coded Multipath Batch Release with Conformal Deadline Certification for Energy-Harvesting Wireless Sensor Networks*.

FRESCO is a controller for clustered, solar-powered wireless sensor networks. It has four parts:

- a zero-order TSK neuro-fuzzy cluster-head elector tuned by the cross-entropy method;
- a factorised graph-attention actor trained with Lagrangian MAPPO, which decides when a cluster head opens a batch (release age τ) and how many repair packets it proposes (r);
- systematic random linear network coding over GF(2^8), split across up to three node-disjoint minimum-ETX paths;
- an adaptive conformal (ACI) shield that raises the repair level, or falls back, whenever the calibrated delay bound for the batch would cross the deadline.

Everything runs in a Python simulator of IEEE 802.15.4 links with duty-cycled radios, CC2420 energy figures and solar harvesting driven by typical-meteorological-year irradiance files. The TMY files ship with `pvlib`, so nothing has to be downloaded.

## Files

All files sit in this one folder. Scripts read and write results in the current directory by default. Set `FRESCO_OUT` to change where results go and `FRESCO_FIG` to change where figures go.

| File | Purpose |
|---|---|
| `fresco_sim.py` | Network simulator: topology, links, energy and harvesting, multipath RLNC release, accounting of age, misses and energy |
| `fresco_agents.py` | Neuro-fuzzy elector, conformal shield, GAT actor and critic, Lagrangian MAPPO, DQN, and the eight baselines |
| `train_all.py` | Training stages: `cem` (elector), `tune` (baseline grids), `eps` (three energy budgets), `rl` (ablations, MAPPO-Penalty, IQL-DQN) |
| `select_eps.py` | Picks the energy budget on the tuning seeds and promotes the selected actor to `actor_fresco.pt` |
| `run_eval.py` | Evaluation on held-out seeds and days: `main abl alpha deadline load harvest scale site shift fail trace` |
| `analyze.py` | Bootstrap CIs, Wilcoxon with Holm correction, rank-biserial effect sizes, Friedman/Nemenyi; writes `analysis.json` |
| `timing.py` | Parameter counts, MACs, per-call latency, shield memory; writes `timing.json` |
| `fig_diagrams.py` | Figs. 1–2 (architecture and training workflow) |
| `make_figures.py` | Figs. 3–9 (results) |
| `make_notebook.py` | Rebuilds `FRESCO_Colab.ipynb` from the source files |
| `FRESCO_Colab.ipynb` | Self-contained notebook for the free Colab CPU runtime |
| `actor_*.pt`, `qnet_dqn.pt`, `elector.npy` | Trained weights used in the paper |
| `eval_*.json`, `trainlog_*.json`, `tuned.json`, `selected_eps.json`, `elector_log.json` | Raw evaluation results, training logs, tuned baseline settings |
| `analysis.json`, `timing.json` | Numbers reported in the tables of the paper |
| `fig*.png` | Figures as they appear in the paper |

## Reproducing

```bash
pip install -r requirements.txt

# use the shipped results: statistics and figures only (a few minutes)
python analyze.py
python timing.py
python fig_diagrams.py
python make_figures.py

# full pipeline from scratch (several hours on two CPU cores)
mkdir fresh && export FRESCO_OUT=fresh FRESCO_FIG=fresh
python train_all.py cem
python train_all.py tune
python train_all.py eps
python select_eps.py
python train_all.py rl
python run_eval.py main abl alpha deadline load harvest scale site shift fail trace
python analyze.py && python timing.py && python fig_diagrams.py && python make_figures.py
```

Each stage skips work whose output file already exists, so an interrupted run can be resumed. Setting `FRESCO_QUICK=1` and `FRESCO_EPISODES=10` gives a short smoke-test run. Its numbers are not the paper's numbers.

On Colab, open `FRESCO_Colab.ipynb`, keep `QUICK = True` for a first check, then set `QUICK = False` for the full protocol.

## Protocol in brief

- 100 nodes, 6 J supercapacitors, a compressed diurnal cycle of 3600 epochs, deadline 30 s, target miscoverage α = 0.05.
- Training uses even TMY days of Greensboro NC and seeds 100–199; baseline tuning and budget selection use seeds 200–203.
- Testing uses 30 seeds (1000–1029) on odd days. Cross-site tests use Miami FL (TMY2) and Sand Point AK (TMY3).
- Baselines: LEACH-Direct, HEED-MMSPEED, FB-MPBatch, SCM-RLNC, EA-AgeThr, Lyapunov-DPP, IQL-DQN, MAPPO-Penalty.

## Scope

All results come from simulation. No hardware testbed was used.

## Licence

MIT (see `LICENSE`).
