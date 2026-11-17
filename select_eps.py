"""Select the energy budget eps_E on the tuning seeds with the same fitness used for baseline tuning,
then promote the chosen policy to actor_fresco.pt."""
import json, os, shutil
import numpy as np
import torch
import fresco_sim as fs
import fresco_agents as fa

torch.set_num_threads(1)
OUT = os.environ.get('FRESCO_OUT', '.')
CANDS = {'fresco_e30': 0.30, 'fresco_e45': 0.45, 'fresco_e60': 0.60}
TUNE_SEEDS = [200, 201, 202, 203]


def main():
    if os.path.exists(f'{OUT}/selected_eps.json'):
        print(open(f'{OUT}/selected_eps.json').read())
        return
    if os.path.exists(f'{OUT}/actor_fresco.pt') and not os.path.exists(f'{OUT}/actor_fresco_e30.pt'):
        shutil.move(f'{OUT}/actor_fresco.pt', f'{OUT}/actor_fresco_e30.pt')
        shutil.move(f'{OUT}/trainlog_fresco.json', f'{OUT}/trainlog_fresco_e30.json')
    el = fa.NeuroFuzzyElector(np.load(f'{OUT}/elector.npy'))
    out = {}
    for name, eps in CANDS.items():
        a = fa.Actor(True)
        a.load_state_dict(torch.load(f'{OUT}/actor_{name}.pt'))
        a.eval()
        rows = [fa.run_episode(fs.default_config(days=list(range(0, 365, 2))), s, fa.MAPPOController(el, a, 'aci', 0.05))
                for s in TUNE_SEEDS]
        out[name] = dict(eps_e=eps, F=float(np.mean([fa.elector_fitness(r, 30.0) for r in rows])),
                         aoi=float(np.mean([r['aoi'] for r in rows])), viol=float(np.mean([r['viol'] for r in rows])),
                         e_bit=float(np.mean([r['e_bit'] for r in rows])),
                         brownout=float(np.mean([r['brownout'] for r in rows])))
        print(name, out[name], flush=True)
    best = max(out, key=lambda k: out[k]['F'])
    shutil.copy(f'{OUT}/actor_{best}.pt', f'{OUT}/actor_fresco.pt')
    shutil.copy(f'{OUT}/trainlog_{best}.json', f'{OUT}/trainlog_fresco.json')
    json.dump(dict(selected=best, eps_e=CANDS[best], candidates=out), open(f'{OUT}/selected_eps.json', 'w'), indent=1)
    print('selected', best)


if __name__ == '__main__':
    main()
