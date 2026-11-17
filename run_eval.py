"""Evaluation of FRESCO, the eight baselines and the ablations on held-out seeds and irradiance days."""
import json, os, sys, time
import multiprocessing as mp
import numpy as np
import torch
import fresco_sim as fs
import fresco_agents as fa

OUT = os.environ.get('FRESCO_OUT', '.')
TEST_DAYS = list(range(1, 365, 2))
TEST_SEEDS = list(range(1000, 1030))
if os.environ.get('FRESCO_QUICK'):
    TEST_SEEDS = list(range(1000, 1005))
TUNED = None


def tuned():
    global TUNED
    if TUNED is None:
        TUNED = json.load(open(f'{OUT}/tuned.json'))['best']
    return TUNED


def _actor(name, gat=True):
    a = fa.Actor(gat)
    a.load_state_dict(torch.load(f'{OUT}/actor_{name}.pt'))
    a.eval()
    return a


def _el():
    return fa.NeuroFuzzyElector(np.load(f'{OUT}/elector.npy'))


def _rule(cname, el):
    C = getattr(fa, cname)
    c = C(el, 'none')
    t = tuned().get(cname)
    if t:
        setattr(c, t['attr'], t['value'])
    return c


def make(name, alpha=0.05):
    torch.set_num_threads(1)
    if name == 'FRESCO':
        return fa.MAPPOController(_el(), _actor('fresco'), 'aci', alpha)
    if name == 'LEACH-Direct':
        c = fa.Immediate(fa.LeachElect(), 'none'); c.name = name; return c
    if name == 'HEED-MMSPEED':
        return _rule('MMSPEEDLike', fa.heed_elect)
    if name == 'FB-MPBatch':
        return _rule('FBMPBatch', fa.fbchs_elect)
    if name == 'SCM-RLNC':
        return _rule('StaticCoded', fa.heed_elect)
    if name == 'EA-AgeThr':
        return _rule('AgeThreshold', fa.heed_elect)
    if name == 'Lyapunov-DPP':
        return _rule('DriftPlusPenalty', fa.heed_elect)
    if name == 'IQL-DQN':
        q = fa.QNet(); q.load_state_dict(torch.load(f'{OUT}/qnet_dqn.pt')); q.eval()
        return fa.DQNController(fa.heed_elect, q)
    if name == 'MAPPO-Penalty':
        return fa.MAPPOController(fa.heed_elect, _actor('mappo_pen', False), 'none', alpha)
    # ablations
    if name == 'w/o NF elector':
        return fa.MAPPOController(fa.heed_elect, _actor('abl_heed'), 'aci', alpha)
    if name == 'w/o RLNC':
        c = fa.MAPPOController(_el(), _actor('abl_nocode'), 'aci', alpha); c.coded = False; return c
    if name == 'w/o learned release':
        return fa.FixedReleaseShielded(_el(), 'aci', alpha)
    if name == 'w/o shield':
        return fa.MAPPOController(_el(), _actor('fresco'), 'none', alpha)
    if name == 'Gaussian bound':
        return fa.MAPPOController(_el(), _actor('fresco'), 'gauss', alpha)
    if name == 'Static split CP':
        return fa.MAPPOController(_el(), _actor('fresco'), 'static', alpha)
    if name == 'w/o GAT':
        return fa.MAPPOController(_el(), _actor('abl_nogat', False), 'aci', alpha)
    if name == 'w/o Lagrangian':
        return fa.MAPPOController(_el(), _actor('abl_nolag'), 'aci', alpha)
    if name == 'FRESCO (seed 2)':
        return fa.MAPPOController(_el(), _actor('fresco_s2'), 'aci', alpha)
    raise KeyError(name)


MAIN = ['FRESCO', 'LEACH-Direct', 'HEED-MMSPEED', 'FB-MPBatch', 'SCM-RLNC', 'EA-AgeThr', 'Lyapunov-DPP',
        'IQL-DQN', 'MAPPO-Penalty']
ABL = ['w/o NF elector', 'w/o RLNC', 'w/o learned release', 'w/o shield', 'Gaussian bound',
       'Static split CP', 'w/o GAT', 'w/o Lagrangian', 'FRESCO (seed 2)']


def _job(job):
    name, cfgkw, seed, alpha, record = job
    cfg = fs.default_config(**cfgkw)
    t0 = time.time()
    ctrl = make(name, alpha)
    r = fa.run_episode(cfg, seed, ctrl, record=record)
    r['sec'] = time.time() - t0
    r['method'] = name
    r['seed'] = seed
    r.update({f'cfg_{k}': v for k, v in cfgkw.items() if k != 'days'})
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}


def run(jobs, tag):
    path = f'{OUT}/eval_{tag}.json'
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    with mp.Pool(2) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    json.dump(res, open(path, 'w'))
    print(tag, len(res), 'episodes', round(time.time() - t0), 's', flush=True)
    return res


def J(methods, seeds, alpha=0.05, record=False, **cfgkw):
    cfgkw.setdefault('days', TEST_DAYS)
    return [(m, cfgkw, s, alpha, record) for m in methods for s in seeds]


if __name__ == '__main__':
    which = sys.argv[1:] or ['main', 'abl', 'alpha', 'deadline', 'load', 'harvest', 'scale', 'site', 'shift', 'fail', 'trace']
    S10 = TEST_SEEDS[:10]
    S15 = TEST_SEEDS[:15]
    KEY = ['FRESCO', 'FB-MPBatch', 'SCM-RLNC', 'EA-AgeThr', 'Lyapunov-DPP']
    if 'main' in which:
        run(J(MAIN, TEST_SEEDS), 'main')
    if 'abl' in which:
        run(J(ABL, TEST_SEEDS), 'abl')
    if 'alpha' in which:
        jobs = []
        for a in [0.01, 0.02, 0.05, 0.1, 0.2]:
            for m in ['FRESCO', 'Static split CP', 'Gaussian bound']:
                ss = TEST_SEEDS if m == 'FRESCO' else S10
                jobs += [(m, dict(days=TEST_DAYS, alpha=a), s, a, False) for s in ss]
        run(jobs, 'alpha')
    if 'deadline' in which:
        jobs = []
        for D in [15.0, 20.0, 45.0, 60.0]:
            jobs += J(KEY, S10, deadline=D)
        run(jobs, 'deadline')
    if 'load' in which:
        jobs = []
        for g in [10.0, 40.0]:
            jobs += J(KEY, S10, gen_period=g)
        run(jobs, 'load')
    if 'harvest' in which:
        jobs = []
        for h in [0.5, 2.0]:
            jobs += J(KEY, S10, harvest_scale=h)
        run(jobs, 'harvest')
    if 'scale' in which:
        jobs = []
        for n in [50, 200, 300]:
            jobs += J(KEY, TEST_SEEDS[:6], N=n)
        run(jobs, 'scale')
    if 'site' in which:
        jobs = []
        for site in ['miami', 'sandpoint']:
            jobs += J(MAIN, S15, site=site, days=list(range(365)))
        run(jobs, 'site')
    if 'shift' in which:
        run(J(['FRESCO', 'Static split CP', 'Gaussian bound', 'w/o shield', 'FB-MPBatch', 'Lyapunov-DPP'], S15,
              extra_loss_db=6.0, shift_step=1800), 'shift')
    if 'fail' in which:
        run(J(MAIN, S15, fail_frac=0.1), 'fail')
    if 'trace' in which:
        jobs = [(m, dict(days=TEST_DAYS, extra_loss_db=6.0, shift_step=1800), 1003, 0.05, True)
                for m in ['FRESCO', 'Static split CP', 'Gaussian bound']]
        jobs += [(m, dict(days=TEST_DAYS), 1003, 0.05, True) for m in ['FRESCO', 'FB-MPBatch']]
        run(jobs, 'trace')
