"""Decision-latency and footprint measurements for FRESCO components (single CPU thread)."""
import json, os, time
import numpy as np
import torch
import fresco_sim as fs
import fresco_agents as fa

torch.set_num_threads(1)
OUT = os.environ.get('FRESCO_OUT', '.')


def macs_actor(gat=True, K=10, nb=4):
    m = 0
    if gat:
        m += fa.N_FEAT * 32 + nb * (2 * 32) + nb * 32   # projection, attention logits, weighted sum
    fin = fa.N_FEAT + (32 if gat else 0)
    m += fin * 64 + 64 * 64 + 64 * (fa.N_TAU + fa.N_R)
    return m


def main():
    res = {}
    actor = fa.Actor(True)
    actor.load_state_dict(torch.load(f'{OUT}/actor_fresco.pt'))
    actor.eval()
    K = 10
    X = torch.randn(1, K, fa.N_FEAT)
    A = torch.ones(1, K, K)
    with torch.no_grad():
        for _ in range(50):
            actor(X, A)
        t0 = time.perf_counter()
        n = 2000
        for _ in range(n):
            actor(X, A)
        res['actor_ms_per_call_K10'] = 1e3 * (time.perf_counter() - t0) / n
    res['actor_params'] = int(sum(p.numel() for p in actor.parameters()))
    res['actor_params_nogat'] = int(sum(p.numel() for p in fa.Actor(False).parameters()))
    res['critic_params'] = int(sum(p.numel() for p in fa.Critic().parameters()))
    res['macs_per_head'] = macs_actor(True)
    res['elector_params'] = fa.NeuroFuzzyElector.dim()
    # elector scoring of 100 nodes
    el = fa.NeuroFuzzyElector(np.load(f'{OUT}/elector.npy'))
    Xn = np.random.rand(100, 4)
    t0 = time.perf_counter()
    for _ in range(200):
        el.score(Xn)
    res['elector_ms_100nodes'] = 1e3 * (time.perf_counter() - t0) / 200
    # shield: quantile computation over 4 strata of W=1500
    sh = fa.ConformalShield(0.05, 'aci')
    for k in range(4):
        sh.scores[k].extend(np.random.lognormal(0, 0.5, 1500))
    t0 = time.perf_counter()
    for _ in range(500):
        sh._qc = None
        sh.q()
    res['shield_ms_per_epoch'] = 1e3 * (time.perf_counter() - t0) / 500
    res['shield_memory_kB'] = 4 * 1500 * 8 / 1024
    # whole-episode wall time of the simulator with FRESCO
    cfg = fs.default_config(days=list(range(1, 365, 2)))
    ctrl = fa.MAPPOController(el, actor, 'aci', 0.05)
    t0 = time.perf_counter()
    fa.run_episode(cfg, 1000, ctrl)
    res['episode_s'] = time.perf_counter() - t0
    # GF(256) RLNC decode cost (Gaussian elimination, B=16, payload 24 B) with numpy tables
    exp = np.zeros(512, np.int64)
    log = np.zeros(256, np.int64)
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= 0x11D
    exp[255:510] = exp[:255]

    def mul(a, b):
        out = exp[(log[a] + log[b]) % 255]
        return np.where((a == 0) | (b == 0), 0, out)

    def inv(a):
        return exp[(255 - log[a]) % 255]

    B, p = 16, 24
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    reps = 50
    for _ in range(reps):
        Mx = rng.integers(0, 256, (B, B + p))
        Mx[:, :B] = rng.integers(1, 256, (B, B))
        for c in range(B):
            piv = next((rr for rr in range(c, B) if Mx[rr, c] != 0), None)
            if piv is None:
                continue
            Mx[[c, piv]] = Mx[[piv, c]]
            Mx[c] = mul(Mx[c], np.full(B + p, inv(Mx[c, c])))
            for rr in range(B):
                if rr != c and Mx[rr, c] != 0:
                    Mx[rr] = Mx[rr] ^ mul(Mx[c], np.full(B + p, Mx[rr, c]))
    res['rlnc_decode_ms_B16'] = 1e3 * (time.perf_counter() - t0) / reps
    res['ctrl_frames_per_round_N100'] = 2 * 100
    json.dump(res, open(f'{OUT}/timing.json', 'w'), indent=1)
    print(res)


if __name__ == '__main__':
    main()
