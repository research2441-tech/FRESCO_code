"""Stage-wise training: (1) CEM for the neuro-fuzzy elector, (2) Lagrangian GAT-MAPPO and the
learning-based ablations/baselines, (3) grid tuning of rule-based baselines on training seeds."""
import json, os, sys, time, pickle
import multiprocessing as mp
import numpy as np
import torch
import fresco_sim as fs
import fresco_agents as fa

OUT = os.environ.get('FRESCO_OUT', '.')
os.makedirs(OUT, exist_ok=True)
TRAIN_DAYS = list(range(0, 365, 2))
TRAIN_SEEDS = list(range(100, 200))
TUNE_SEEDS = [200, 201, 202, 203]
EPISODES = int(os.environ.get('FRESCO_EPISODES', 70))


def base_cfg(**kw):
    return fs.default_config(days=TRAIN_DAYS, **kw)


def stage_cem():
    path = f'{OUT}/elector.npy'
    if os.path.exists(path):
        return
    log = []
    t0 = time.time()
    with mp.Pool(2) as pool:
        mean_el, best_el = fa.train_elector_cem(base_cfg(), TRAIN_SEEDS, iters=8, pop=16, elite=4, log=log, pool=pool)
    np.save(path, mean_el.theta)
    json.dump(dict(log=log, sec=time.time() - t0), open(f'{OUT}/elector_log.json', 'w'))
    print('CEM done', time.time() - t0, log[-1], flush=True)


def load_elector():
    return fa.NeuroFuzzyElector(np.load(f'{OUT}/elector.npy'))


def _eps_e():
    p = f'{OUT}/selected_eps.json'
    return json.load(open(p))['eps_e'] if os.path.exists(p) else float(os.environ.get('FRESCO_EPS_E', 0.3))
VARIANTS = {
    # name: (elector, use_gat, lagrangian, coded, shield)
    'fresco_e45':    ('nf', True, True, True, 'aci'),
    'fresco_e60':    ('nf', True, True, True, 'aci'),
    'fresco':        ('nf', True, True, True, 'aci'),
    'fresco_s2':     ('nf', True, True, True, 'aci'),
    'abl_nogat':     ('nf', False, True, True, 'aci'),
    'abl_nolag':     ('nf', True, False, True, 'aci'),
    'abl_nocode':    ('nf', True, True, False, 'aci'),
    'abl_heed':      ('heed', True, True, True, 'aci'),
    'mappo_pen':     ('heed', False, False, True, 'none'),
}


def train_variant(name):
    torch.set_num_threads(1)
    path = f'{OUT}/actor_{name}.pt'
    if os.path.exists(path):
        return name
    el_kind, gat, lag, coded, shield = VARIANTS[name]
    el = load_elector() if el_kind == 'nf' else fa.heed_elect
    log = []
    t0 = time.time()
    ts = 1 if name.endswith('_s2') else 0
    eps_e = {'fresco_e20': 0.2, 'fresco_e45': 0.45, 'fresco_e60': 0.6}.get(name, _eps_e())
    actor, critic, learner = fa.train_mappo(base_cfg(), el, TRAIN_SEEDS, use_gat=gat, lagrangian=lag, coded=coded,
                                            shield_mode=shield, episodes=EPISODES, torch_seed=ts, eps_e=eps_e,
                                            log=log)
    torch.save(actor.state_dict(), path)
    json.dump(dict(log=log, sec=time.time() - t0), open(f'{OUT}/trainlog_{name}.json', 'w'))
    print('trained', name, round(time.time() - t0), log[-1], flush=True)
    return name


def train_dqn_job(_):
    torch.set_num_threads(1)
    path = f'{OUT}/qnet_dqn.pt'
    if os.path.exists(path):
        return 'dqn'
    log = []
    t0 = time.time()
    q = fa.train_dqn(base_cfg(), fa.heed_elect, TRAIN_SEEDS, episodes=40, log=log)
    torch.save(q.state_dict(), path)
    json.dump(dict(log=log, sec=time.time() - t0), open(f'{OUT}/trainlog_dqn.json', 'w'))
    print('trained dqn', round(time.time() - t0), log[-1], flush=True)
    return 'dqn'


GRIDS = {
    'MMSPEEDLike': ('rho', [0.9, 0.95, 0.99]),
    'FBMPBatch': ('timeout', [5.0, 10.0, 20.0, 30.0]),
    'StaticCoded': ('timeout', [3.0, 5.0, 10.0, 15.0, 20.0]),
    'AgeThreshold': ('tau0', [2.0, 3.0, 5.0, 10.0, 15.0]),
    'DriftPlusPenalty': ('V', [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]),
}
ELECT = {'MMSPEEDLike': 'heed', 'FBMPBatch': 'fbchs', 'StaticCoded': 'heed', 'AgeThreshold': 'heed', 'DriftPlusPenalty': 'heed'}


def _tune_eval(job):
    cname, attr, val = job
    C = getattr(fa, cname)
    el = {'heed': fa.heed_elect, 'fbchs': fa.fbchs_elect}[ELECT[cname]]
    fit = []
    for s in TUNE_SEEDS:
        ctrl = C(el, 'none')
        setattr(ctrl, attr, val)
        r = fa.run_episode(base_cfg(), s, ctrl)
        fit.append(fa.elector_fitness(r, 30.0))
    return cname, val, float(np.mean(fit))


def stage_tune():
    path = f'{OUT}/tuned.json'
    if os.path.exists(path):
        return
    jobs = [(c, a, v) for c, (a, vals) in GRIDS.items() for v in vals]
    with mp.Pool(2) as pool:
        res = pool.map(_tune_eval, jobs)
    best = {}
    for c, v, f in res:
        if c not in best or f > best[c][1]:
            best[c] = (v, f)
    json.dump(dict(best={c: dict(attr=GRIDS[c][0], value=v, fit=f) for c, (v, f) in best.items()},
                   all=[list(r) for r in res]), open(path, 'w'), indent=1)
    print('tuned', best, flush=True)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('cem', 'all'):
        stage_cem()
    if stage in ('tune', 'all'):
        stage_tune()
    if stage == 'eps':
        with mp.Pool(2) as pool:
            pool.map(train_variant, ['fresco', 'fresco_e45', 'fresco_e60'])
    if stage in ('rl', 'all'):
        names = [n for n in VARIANTS if not n.startswith('fresco_e')]
        with mp.Pool(2) as pool:
            jobs = [pool.apply_async(train_variant, (n,)) for n in names] + [pool.apply_async(train_dqn_job, (0,))]
            for j in jobs:
                j.get()
