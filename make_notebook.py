"""Assemble the Colab notebook from the source files of the release."""
import nbformat as nbf

FILES = ['fresco_sim.py', 'fresco_agents.py', 'train_all.py', 'select_eps.py', 'run_eval.py', 'analyze.py',
         'timing.py', 'fig_diagrams.py', 'make_figures.py']

nb = nbf.v4.new_notebook()
cells = []
cells.append(nbf.v4.new_markdown_cell(
    "# FRESCO: freshness-aware coded multipath routing with conformally calibrated batch release\n\n"
    "This notebook reproduces every number, table and figure of the manuscript on the free tier of Google Colab "
    "(CPU runtime is enough). It writes the source files, trains the neuro-fuzzy elector and the Lagrangian "
    "GAT-MAPPO release policy, tunes and runs the eight baselines, evaluates on held-out topologies and irradiance "
    "days, computes the statistics and draws the figures.\n\n"
    "* `QUICK = True` runs a reduced pipeline (10 training episodes, 5 test episodes) to check that "
    "the pipeline works end to end; its numbers are not the paper's numbers.\n"
    "* `QUICK = False` runs the full protocol of the paper; on the two vCPUs of a free runtime "
    "it takes several hours, so keep the runtime alive or run the stages in separate sessions (every stage "
    "skips work whose output file already exists).\n\n"
    "Irradiance comes from the typical-meteorological-year files shipped inside the `pvlib` package "
    "(Greensboro NC, Miami FL, Sand Point AK), so no download is needed."))
cells.append(nbf.v4.new_code_cell("!pip -q install pvlib==0.16.1 nbformat\nimport torch, pvlib, numpy, scipy\nprint(torch.__version__, pvlib.__version__)"))
cells.append(nbf.v4.new_code_cell("QUICK = True  # set to False for the full protocol of the paper\n"
                                  "import os\nos.environ['FRESCO_OUT'] = 'runs'; os.environ['FRESCO_FIG'] = 'figs'\n"
                                  "os.environ['FRESCO_EPISODES'] = '10' if QUICK else '70'\n"
                                  "if QUICK:\n    os.environ['FRESCO_QUICK'] = '1'\n"
                                  "os.makedirs('runs', exist_ok=True); os.makedirs('figs', exist_ok=True)"))
for f in FILES:
    src = open(f).read()
    cells.append(nbf.v4.new_markdown_cell(f"### `{f}`"))
    cells.append(nbf.v4.new_code_cell(f"%%writefile {f}\n" + src))
cells.append(nbf.v4.new_markdown_cell("## 1. Stage 1: neuro-fuzzy elector (cross-entropy method) and baseline tuning"))
cells.append(nbf.v4.new_code_cell("!python train_all.py cem\n!python train_all.py tune"))
cells.append(nbf.v4.new_markdown_cell("## 2. Stage 2: Lagrangian GAT-MAPPO for three energy budgets, selection on tuning seeds"))
cells.append(nbf.v4.new_code_cell("!python train_all.py eps\n!python select_eps.py"))
cells.append(nbf.v4.new_markdown_cell("## 3. Ablation variants and learned baselines"))
cells.append(nbf.v4.new_code_cell("!python train_all.py rl"))
cells.append(nbf.v4.new_markdown_cell("## 4. Evaluation on held-out seeds, days and sites"))
cells.append(nbf.v4.new_code_cell("!python run_eval.py main abl alpha deadline load harvest scale site shift fail trace"))
cells.append(nbf.v4.new_markdown_cell("## 5. Statistics, timing and figures"))
cells.append(nbf.v4.new_code_cell("!python analyze.py\n!python timing.py\n!python fig_diagrams.py\n!python make_figures.py"))
cells.append(nbf.v4.new_code_cell(
    "import json, pandas as pd\nA = json.load(open('runs/analysis.json'))\n"
    "rows = {m: {k: v[k][0] for k in ['aoi','paoi95','viol','miss_released','pdr','e_bit','brownout']} for m, v in A['main'].items()}\n"
    "pd.DataFrame(rows).T.round(3)"))
cells.append(nbf.v4.new_code_cell(
    "from IPython.display import Image, display\nimport glob\nfor f in sorted(glob.glob('figs/*.png')):\n    print(f); display(Image(f, width=700))"))
nb['cells'] = cells
nb['metadata'] = {'colab': {'provenance': []}, 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                  'language_info': {'name': 'python'}}
nbf.write(nb, 'FRESCO_Colab.ipynb')
print('written', len(cells), 'cells')
