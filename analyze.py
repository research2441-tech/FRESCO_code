"""Statistics for the FRESCO evaluation: bootstrap CIs, Wilcoxon-Holm, rank-biserial, Friedman/Nemenyi."""
import json, os, sys
import numpy as np
from scipy import stats

OUT = os.environ.get('FRESCO_OUT', '.')
METRICS = ['aoi', 'paoi95', 'viol', 'miss_released', 'miss_feasible', 'feasible_frac', 'miss_feas_share', 'pdr', 'plr', 'e_bit', 'brownout', 'mean_batch',
           'infeasible', 'shield_raise', 'miscover', 'lost_mc']
LOWER_BETTER = {'aoi': True, 'paoi95': True, 'viol': True, 'miss_released': True, 'pdr': False,
                'e_bit': True, 'brownout': True}


def load(tag):
    p = f'{OUT}/eval_{tag}.json'
    return json.load(open(p)) if os.path.exists(p) else []


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def by_method(rows, key='method'):
    d = {}
    for r in rows:
        d.setdefault(r[key], []).append(r)
    for k in d:
        d[k].sort(key=lambda r: r['seed'])
    return d


def summarize(rows):
    out = {}
    for m, rs in by_method(rows).items():
        out[m] = {k: boot_ci([r.get(k, np.nan) for r in rs]) for k in METRICS}
        out[m]['n'] = len(rs)
    return out


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def rank_biserial(diff):
    d = np.asarray(diff, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    rk = stats.rankdata(np.abs(d))
    wp, wm = rk[d > 0].sum(), rk[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def paired_tests(rows, ref, metrics=('aoi', 'viol', 'e_bit', 'paoi95', 'brownout')):
    d = by_method(rows)
    others = [m for m in d if m != ref]
    res = {}
    for met in metrics:
        a = np.array([r[met] for r in d[ref]])
        ps, rbs, deltas = [], [], []
        for m in others:
            b = np.array([r[met] for r in d[m]])
            diff = a - b
            if np.allclose(diff, 0):
                p = 1.0
            else:
                p = float(stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided').pvalue)
            ps.append(p)
            rbs.append(rank_biserial(diff))
            deltas.append(float(np.mean(diff)))
        adj = holm(ps)
        res[met] = {m: dict(p=ps[i], p_holm=float(adj[i]), r_rb=rbs[i], delta=deltas[i]) for i, m in enumerate(others)}
    return res


def friedman(rows, metric='aoi'):
    d = by_method(rows)
    ms = list(d)
    X = np.array([[r[metric] for r in d[m]] for m in ms]).T  # episodes x methods
    stat, p = stats.friedmanchisquare(*X.T)
    ranks = np.array([stats.rankdata(row) for row in X])
    mr = ranks.mean(0)
    k, n = len(ms), X.shape[0]
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}[k]
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return dict(stat=float(stat), p=float(p), mean_rank={m: float(r) for m, r in zip(ms, mr)}, cd=float(cd), n=n, k=k)


def sweep(rows, param):
    out = {}
    for r in rows:
        key = (r['method'], r.get(f'cfg_{param}'))
        out.setdefault(key, []).append(r)
    res = {}
    for (m, v), rs in out.items():
        res.setdefault(m, {})[str(v)] = {k: boot_ci([x[k] for x in rs]) for k in ['aoi', 'viol', 'miss_released', 'miss_feasible', 'feasible_frac', 'miss_feas_share', 'e_bit', 'brownout', 'pdr', 'miscover', 'paoi95']}
    return res


if __name__ == '__main__':
    R = {}
    main = load('main')
    if main:
        R['main'] = summarize(main)
        R['tests'] = paired_tests(main, 'FRESCO')
        R['friedman_aoi'] = friedman(main, 'aoi')
        R['friedman_viol'] = friedman(main, 'viol')
    abl = load('abl')
    if abl and main:
        full = [r for r in main if r['method'] == 'FRESCO']
        R['abl'] = summarize(abl + full)
        R['abl_tests'] = paired_tests(abl + full, 'FRESCO', metrics=('aoi', 'viol', 'e_bit', 'miss_released', 'brownout'))
    for tag, par in [('deadline', 'deadline'), ('load', 'gen_period'), ('harvest', 'harvest_scale'), ('scale', 'N'), ('alpha', 'alpha')]:
        rows = load(tag)
        if rows:
            R['sweep_' + tag] = sweep(rows, par)
    for tag in ['site', 'shift', 'fail']:
        rows = load(tag)
        if rows:
            if tag == 'site':
                for site in ['miami', 'sandpoint']:
                    rr = [r for r in rows if r.get('cfg_site') == site]
                    R[f'site_{site}'] = summarize(rr)
                    R[f'site_{site}_tests'] = paired_tests(rr, 'FRESCO', metrics=('aoi', 'viol', 'e_bit'))
            else:
                R[tag] = summarize(rows)
                if tag == 'fail':
                    R['fail_tests'] = paired_tests(rows, 'FRESCO', metrics=('aoi', 'viol', 'e_bit'))
    json.dump(R, open(f'{OUT}/analysis.json', 'w'), indent=1)
    if 'main' in R:
        for m, v in R['main'].items():
            print(f"{m:16s} " + ' '.join(f"{k}={v[k][0]:.3f}" for k in ['aoi', 'paoi95', 'viol', 'miss_released', 'pdr', 'e_bit', 'brownout', 'mean_batch']))
        print(R['friedman_aoi'])
