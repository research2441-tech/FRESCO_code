"""FRESCO packet-level simulator for clustered, energy-harvesting IEEE 802.15.4 sensor networks.

One simulation step equals DT seconds. A full diurnal irradiance cycle (24 h of TMY data) is
compressed into DAY_STEPS steps, and storage capacity is scaled with the same factor, so that the
ratio between night-time drain and capacity matches an uncompressed deployment.
"""
import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

# ----------------------------------------------------------------------------- constants
DT = 1.0                 # decision epoch (s)
DAY_STEPS = 3600         # one compressed diurnal cycle
ROUND = 300              # cluster-head election period (steps)
V_SUP = 3.0              # supply voltage (V)
P_TX = 17.4e-3 * V_SUP   # CC2420 transmit, 0 dBm (W)
P_RX = 18.8e-3 * V_SUP   # CC2420 receive (W)
BITRATE = 250e3          # 802.15.4 O-QPSK 2.4 GHz
HDR_BYTES = 17           # MAC header + FCS + PHY header
PAYLOAD = 24             # sensing update payload (bytes)
T_W = 2.0                # relay wake-up interval for low-power listening (s)
STROBE = 0.2             # mean phase-locked strobe duration per hop burst (s)
T_W_CH = 0.25            # cluster-head wake-up interval (s)
MAX_ATT = 3              # link-layer attempts per frame
P_LISTEN_MEMBER = 0.15e-3
P_LISTEN_RELAY = 0.15e-3
P_LISTEN_CH = 3.0e-3
CAP_J = 6.0              # storage capacity (J), compressed-day units
PANEL = 1.0e-4 * 0.15    # 1 cm^2 panel, 15 % efficiency  -> W per (W/m^2)
B_MAX = 16               # cluster-head buffer limit
R_LEVELS = (0, 1, 2, 3)  # repair-packet levels
N_ACT = 1 + len(R_LEVELS)
REVIVE = 0.10            # storage fraction needed to restart after brown-out
PL0, ETA, SIG_SH = 55.0, 3.0, 4.0
NOISE_DBM, PT_DBM = -100.0, 0.0
FADE_RHO, FADE_SIG = 0.995, 1.5

DATA_DIR = None


def _tmy_dir():
    import pvlib
    return os.path.join(os.path.dirname(pvlib.__file__), 'data')


_GHI_CACHE = {}


def load_ghi(site):
    """Hourly GHI (W/m^2), 365 x 24, from the TMY files distributed with pvlib."""
    if site in _GHI_CACHE:
        return _GHI_CACHE[site]
    from pvlib.iotools import read_tmy3, read_tmy2
    d = _tmy_dir()
    if site == 'greensboro':
        df, _ = read_tmy3(os.path.join(d, '723170TYA.CSV'), map_variables=True)
        g = df['ghi'].values
    elif site == 'sandpoint':
        df, _ = read_tmy3(os.path.join(d, '703165TY.csv'), map_variables=True)
        g = df['ghi'].values
    elif site == 'miami':
        df, _ = read_tmy2(os.path.join(d, '12839.tm2'))
        g = df['GHI'].values
    else:
        raise ValueError(site)
    g = np.asarray(g, float).reshape(365, 24)
    _GHI_CACHE[site] = g
    return g


def prr_oqpsk(snr_db, frame_bytes):
    """Packet reception ratio of IEEE 802.15.4 O-QPSK (bit error expression of the standard)."""
    snr = 10.0 ** (np.asarray(snr_db, float) / 10.0)
    ber = np.zeros_like(snr)
    from math import comb
    for k in range(2, 17):
        ber = ber + ((-1) ** k) * comb(16, k) * np.exp(20.0 * snr * (1.0 / k - 1.0))
    ber = np.clip(ber * (8.0 / 15.0) * (1.0 / 16.0), 0.0, 0.5)
    return (1.0 - ber) ** (8 * frame_bytes)


def frame_time(nbytes):
    return 8.0 * (nbytes + 6) / BITRATE


class Config(dict):
    def __getattr__(self, k):
        return self[k]


def default_config(**kw):
    c = Config(N=100, density=100.0 / 100.0 ** 2, gen_period=20.0, deadline=30.0, alpha=0.05,
               site='greensboro', days=None, harvest_scale=1.0, extra_loss_db=0.0, shift_step=None,
               fail_frac=0.0, steps=DAY_STEPS, coding=True, k_paths=3)
    c.update(kw)
    return c


# ----------------------------------------------------------------------------- environment
class WSN:
    """Clustered energy-harvesting WSN with duty-cycled multi-hop multipath delivery."""

    def __init__(self, cfg, seed):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        rng = self.rng
        N = cfg.N
        self.N = N
        L = np.sqrt(N / cfg.density)
        self.L = L
        self.pos = rng.uniform(0, L, size=(N, 2))
        self.sink_pos = np.array([L / 2.0, -5.0])
        P = np.vstack([self.pos, self.sink_pos])
        self.S = N  # sink index
        d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, 1.0)
        self.dist = d
        sh = rng.normal(0, SIG_SH, size=(N + 1, N + 1))
        sh = (sh + sh.T) / np.sqrt(2.0)
        self.snr0 = PT_DBM - (PL0 + 10 * ETA * np.log10(np.maximum(d, 1.0)) + sh) - NOISE_DBM
        np.fill_diagonal(self.snr0, -50.0)
        fb = HDR_BYTES + PAYLOAD + B_MAX
        self.prr0 = prr_oqpsk(self.snr0, fb)
        self.adj = self.prr0 >= 0.3
        np.fill_diagonal(self.adj, False)
        # estimated link quality (beacon-based estimate with noise), refined from traffic
        self.prr_hat = np.clip(self.prr0 + rng.normal(0, 0.05, self.prr0.shape), 0.05, 1.0) * self.adj
        self.fade = np.zeros(N + 1)
        self.shade = rng.uniform(0.3, 1.0, size=N)
        # irradiance profile for this episode (compressed day)
        ghi = load_ghi(cfg.site)
        days = cfg.days if cfg.days is not None else np.arange(365)
        self.day = int(rng.choice(days))
        prof = ghi[self.day]
        t_h = np.arange(cfg.steps) * 24.0 / DAY_STEPS
        self.ghi_t = np.interp(t_h, np.arange(24) + 0.5, prof, period=24)
        self.store = rng.uniform(0.5, 0.8, size=N) * CAP_J
        self.alive = np.ones(N, bool)
        self.failed = np.zeros(N, bool)
        if cfg.fail_frac > 0:
            idx = rng.choice(N, int(round(cfg.fail_frac * N)), replace=False)
            self.fail_at = dict(zip(idx, rng.integers(cfg.steps // 4, 3 * cfg.steps // 4, size=len(idx))))
        else:
            self.fail_at = {}
        self.harv_ewma = np.zeros(N)
        self.relay_load = np.zeros(N)
        self.busy = np.zeros(N + 1)
        # traffic
        self.phase = rng.uniform(0, cfg.gen_period, size=N)
        self.next_gen = self.phase.copy()
        self.U = np.zeros(N)          # generation time of freshest delivered update
        self.t = 0
        self.is_ch = np.zeros(N, bool)
        self.chs = np.array([], int)
        self.member_of = -np.ones(N, int)
        self.buffers = {}
        self.paths = {}
        self.deliv = {}               # step -> list of (src, gen)
        self._cache = {}
        # metrics
        self.m = dict(gen=0, deliv=0, ontime=0, aoi_sum=0.0, aoi_n=0, energy=0.0, harvest=0.0,
                      bits=0.0, dead_frac=0.0, peak=[], batches=0, ctrl_frames=0, lost_mc=0,
                      released=0, lost_dead=0, lost_hop=0, shield_raise=0, infeasible=0, per_ch_ontime=[])
        self.peak_track = np.zeros(N)
        self.last_U = np.zeros(N)

    # ------------------------------------------------------------------ link/energy helpers
    def link_prr(self, i, j):
        snr = self.snr0[i, j] - self.fade[i] - self.fade[j] - self.cfg.extra_loss_db * self._shift_on()
        return prr_oqpsk(snr, HDR_BYTES + PAYLOAD + B_MAX)

    def _shift_on(self):
        s = self.cfg.shift_step
        return 1.0 if (s is not None and self.t >= s) else 0.0

    def spend(self, i, e):
        if i == self.S:
            return True
        if not self.alive[i]:
            return False
        self.store[i] -= e
        self.m['energy'] += e
        if self.store[i] <= 0:
            self.store[i] = 0.0
            self.alive[i] = False
            return False
        return True

    # ------------------------------------------------------------------ routing
    def compute_paths(self, src, k=3):
        """Up to k node-disjoint min-ETX paths from src to sink on the estimated link graph."""
        N1 = self.N + 1
        ok = np.append(self.alive & ~self.failed, True)
        w = np.where(self.adj & (self.prr_hat > 0.05), 1.0 / np.maximum(self.prr_hat, 1e-3), 0.0)
        w = w * ok[:, None] * ok[None, :]
        removed = np.zeros(N1, bool)
        out = []
        for _ in range(k):
            ww = w.copy()
            ww[removed, :] = 0
            ww[:, removed] = 0
            g = csr_matrix(ww)
            dist, pred = dijkstra(g, directed=True, indices=src, return_predecessors=True)
            if not np.isfinite(dist[self.S]):
                break
            p = [self.S]
            while p[-1] != src:
                p.append(pred[p[-1]])
            p = p[::-1]
            out.append(p)
            for v in p[1:-1]:
                removed[v] = True
            if len(p) == 2:  # direct link cannot be reused disjointly
                w[src, self.S] = 0
        return out

    def path_stats(self, p):
        """Nominal (estimated) per-packet end-to-end success and mean delay of a path."""
        key = ('p',) + tuple(p)
        if key in self._cache:
            return self._cache[key]
        q = 1.0
        etx = 0.0
        for a, b in zip(p[:-1], p[1:]):
            ph = self.prr_hat[a, b]
            q *= 1 - (1 - ph) ** MAX_ATT
            etx += 1.0 / max(ph, 1e-3)
        hops = len(p) - 1
        self._cache[key] = (q, hops, etx)
        return q, hops, etx

    # ------------------------------------------------------------------ clustering
    def node_features(self):
        """Inputs of the cluster-head elector, all scaled to [0, 1]."""
        e = self.store / CAP_J
        h = np.clip(self.harv_ewma / (PANEL * 1000.0), 0, 1)
        nb = self.adj[:self.N, :self.N]
        etx = np.where(nb, 1.0 / np.maximum(self.prr_hat[:self.N, :self.N], 1e-3), 0).sum(1) / np.maximum(nb.sum(1), 1)
        x = np.clip((etx - 1.0) / 2.0, 0, 1)
        dsink = np.clip(self.dist[:self.N, self.S] / (self.L * 1.2), 0, 1)
        q = np.clip(self.relay_load / (np.max(self.relay_load) + 1e-9), 0, 1)
        return np.stack([e, h, x, q], 1), dsink

    def elect(self, scores, K):
        """Greedy election of K heads by score with spatial suppression."""
        cand = np.where(self.alive & ~self.failed & (self.store > 0.15 * CAP_J))[0]
        if len(cand) == 0:
            cand = np.where(self.alive & ~self.failed)[0]
        rmin = 0.55 * self.L / np.sqrt(K)
        order = cand[np.argsort(-scores[cand])]
        chosen = []
        for i in order:
            if all(self.dist[i, j] >= rmin for j in chosen):
                chosen.append(i)
            if len(chosen) == K:
                break
        if len(chosen) < K:
            for i in list(order) + list(np.argsort(-self.store)):
                if i not in chosen:
                    chosen.append(int(i))
                if len(chosen) == K:
                    break
        return np.array(chosen, int)

    def set_heads(self, chs):
        old_buffers = self.buffers
        self.is_ch[:] = False
        self.is_ch[chs] = True
        self.chs = np.array(chs, int)
        self.buffers = {int(c): old_buffers.get(int(c), []) for c in chs}
        self.affiliate()
        self.paths = {int(c): self.compute_paths(int(c), self.cfg.k_paths) for c in chs}
        self._cache = {}
        # control overhead: HELLO from every node, head announcement, join
        n_al = int(self.alive.sum())
        frames = n_al + len(chs) + max(n_al - len(chs), 0)
        self.m['ctrl_frames'] += frames
        ft = frame_time(HDR_BYTES + 8)
        for i in np.where(self.alive)[0]:
            self.spend(i, P_TX * (ft + 0.01) * 2)
        self.relay_load *= 0.0

    def affiliate(self):
        alive_ch = [c for c in self.chs if self.alive[c]]
        self.member_of[:] = -1
        if not alive_ch:
            return
        ac = np.array(alive_ch)
        etx = np.where(self.adj[:self.N][:, ac], 1.0 / np.maximum(self.prr_hat[:self.N][:, ac], 1e-3), 1e6)
        etx = etx + 0.02 * self.dist[:self.N][:, ac]
        self.member_of = ac[np.argmin(etx, 1)]
        self.member_of[ac] = ac

    # ------------------------------------------------------------------ batch release
    def allocate(self, paths, B, r):
        """Inner allocation: spread B+r packets over the disjoint paths in proportion to
        w_j = q_j / h_j (per-packet success over hop count), largest-remainder rounding.
        Without repair packets the whole batch takes the best path."""
        stats = [self.path_stats(p) for p in paths]
        n = np.zeros(len(paths), int)
        tot = B + r
        if r == 0 or len(paths) == 1:
            n[0] = tot
            return n, stats
        w = np.array([s[0] / s[1] for s in stats])
        w = w / w.sum()
        raw = w * tot
        n = np.floor(raw).astype(int)
        rem = tot - n.sum()
        if rem > 0:
            n[np.argsort(-(raw - n))[:rem]] += 1
        return n, stats

    @staticmethod
    def _p_decode(n, stats, B):
        dist = np.zeros(int(n.sum()) + 1)
        dist[0] = 1.0
        for j, nj in enumerate(n):
            if nj == 0:
                continue
            q = stats[j][0]
            from math import comb
            pmf = np.array([comb(nj, k) * q ** k * (1 - q) ** (nj - k) for k in range(nj + 1)])
            dist = np.convolve(dist, pmf)[:len(dist)]
        return dist[B:].sum()

    def nominal_delay(self, n, stats):
        d = 0.0
        for nj, (q, hops, etx) in zip(n, stats):
            if nj > 0:
                d = max(d, hops * (T_W / 2.0 + STROBE) + nj * frame_time(HDR_BYTES + PAYLOAD + B_MAX) * etx)
        return max(d, 1e-3)

    def release(self, ch, r, coded=True):
        """Release the head's buffer as one batch with r repair packets. Returns batch record."""
        buf = self.buffers[ch]
        if not buf:
            return None
        paths = [p for p in self.paths.get(ch, []) if all(v == self.S or self.alive[v] for v in p[1:])]
        if not paths:
            self.paths[ch] = self.compute_paths(ch, self.cfg.k_paths)
            paths = self.paths[ch]
        B = len(buf)
        # nominal delay as seen by the deadline check of this epoch (same cached estimates)
        d_nom = self.nominal_all(ch)[0] if r in R_LEVELS else None
        self.buffers[ch] = []
        e0 = self.m['energy']
        self.m['batches'] += 1
        self.m['released'] += B
        gens = np.array([g for (_, g) in buf])
        srcs = np.array([s for (s, _) in buf])
        if not paths or not self.alive[ch]:
            return dict(B=B, T_c=np.inf, d_hat=1.0, age=self.t - gens.min(), srcs=srcs, gens=gens, r=r,
                        deliver_t=np.full(B, np.inf), energy=0.0)
        n, stats = self.allocate(paths, B, r)
        d_hat = self.nominal_delay(n, stats)
        if d_nom is not None and np.isfinite(d_nom[R_LEVELS.index(r)]):
            d_hat = float(d_nom[R_LEVELS.index(r)])
        nbytes = HDR_BYTES + PAYLOAD + (B if coded and r > 0 else 0)
        ft = frame_time(nbytes)
        # packet ids: 0..B-1 systematic originals, B..B+r-1 repairs
        ids = np.arange(B + r)
        assign = np.repeat(np.arange(len(paths)), n)
        order = np.argsort([-stats[j][0] for j in assign], kind='stable')
        assign = assign[order]  # systematic packets go to the most reliable paths first
        arrivals = []  # (time, packet ids)
        for j, p in enumerate(paths):
            pk = ids[assign == j]
            if len(pk) == 0:
                continue
            tcur = 0.0
            surv = pk.copy()
            for a, b in zip(p[:-1], p[1:]):
                if len(surv) == 0:
                    break
                ph = self.link_prr(a, b)
                att = np.zeros(len(surv), int)
                ok = np.zeros(len(surv), bool)
                for _ in range(MAX_ATT):
                    m = ~ok
                    att[m] += 1
                    ok[m] = self.rng.random(m.sum()) < ph
                self.prr_hat[a, b] = 0.97 * self.prr_hat[a, b] + 0.03 * (ok.sum() / max(att.sum(), 1))
                start = max(self.t + tcur, self.busy[a])
                wait = self.rng.uniform(0, T_W) if b != self.S else self.rng.uniform(0, 0.05)
                dur = STROBE + att.sum() * ft
                e_tx = P_TX * dur
                if not self.spend(a, e_tx):
                    self.m['lost_dead'] += len(surv)
                    surv = surv[:0]
                    break
                if b != self.S:
                    if not self.spend(b, P_RX * (ok.sum() * ft + 0.01)):
                        self.m['lost_dead'] += len(surv)
                        surv = surv[:0]
                        break
                    self.relay_load[b] += 1
                self.busy[a] = start + wait + dur
                tcur = (start - self.t) + wait + dur
                self.m['lost_hop'] += int((~ok).sum())
                surv = surv[ok]
            if len(surv):
                arrivals.append((tcur, surv))
        # decoding at the sink
        deliver_t = np.full(B, np.inf)
        arrivals.sort(key=lambda z: z[0])
        cnt = 0
        dec_t = np.inf
        got = set()
        for tt, pk in arrivals:
            for pid in pk:
                if pid < B:
                    deliver_t[pid] = min(deliver_t[pid], tt)
                got.add(int(pid))
            cnt = len(got)
            if coded and cnt >= B and dec_t == np.inf:
                dec_t = tt
        if not coded:
            # repairs are duplicates of originals, assigned cyclically
            for tt, pk in arrivals:
                for pid in pk:
                    if pid >= B:
                        o = (pid - B) % B
                        deliver_t[o] = min(deliver_t[o], tt)
        if coded:
            deliver_t = np.minimum(deliver_t, dec_t)
        T_c = deliver_t.max()
        for k in range(B):
            if np.isfinite(deliver_t[k]):
                st = self.t + int(np.ceil(deliver_t[k] / DT))
                self.deliv.setdefault(st, []).append((srcs[k], gens[k]))
                self.m['bits'] += 8 * PAYLOAD
                self.m['deliv'] += 1
                if (self.t + deliver_t[k]) - gens[k] <= self.cfg.deadline:
                    self.m['ontime'] += 1
        self._cache = {k: v for k, v in self._cache.items() if k[0] != ch and k[0] != 'p'}
        return dict(B=B, T_c=T_c, d_hat=d_hat, age=self.t - gens.min(), srcs=srcs, gens=gens, r=r, n=n,
                    deliver_t=deliver_t, energy=self.m['energy'] - e0)

    def nominal_all(self, ch):
        """Nominal completion delay d_hat(r) and decoding probability for every repair level (cached)."""
        buf = self.buffers.get(ch, [])
        B = max(len(buf), 1)
        key = (ch, B)
        if key in self._cache:
            return self._cache[key]
        paths = [p for p in self.paths.get(ch, []) if all(v == self.S or self.alive[v] for v in p[1:])]
        if not paths:
            res = (np.full(len(R_LEVELS), np.inf), np.zeros(len(R_LEVELS)))
        else:
            dh = []
            for r in R_LEVELS:
                n, stats = self.allocate(paths, B, r)
                dh.append(self.nominal_delay(n, stats))
            res = (np.array(dh), None)
        self._cache[key] = res
        return res

    def predict_bounds(self, ch, q_ratio):
        """Calibrated completion-time bound q_r * d_hat(r); q_ratio is a scalar or one value per level."""
        dh, _ = self.nominal_all(ch)
        q = np.broadcast_to(np.asarray(q_ratio, float), dh.shape)
        with np.errstate(invalid='ignore'):
            U = np.where(np.isfinite(dh), q * dh, np.inf)
        return U, dh

    # ------------------------------------------------------------------ per-step dynamics
    def step_physics(self):
        cfg = self.cfg
        t = self.t
        # failures
        for i, ft_ in list(self.fail_at.items()):
            if t >= ft_:
                self.failed[i] = True
                self.alive[i] = False
                del self.fail_at[i]
        # harvesting
        ph = PANEL * self.ghi_t[min(t, len(self.ghi_t) - 1)] * self.shade * cfg.harvest_scale
        self.harv_ewma = 0.98 * self.harv_ewma + 0.02 * ph
        self.store = np.minimum(self.store + ph * DT * ~self.failed, CAP_J)
        self.m['harvest'] += float((ph * DT).sum())
        revive = (~self.alive) & (~self.failed) & (self.store > REVIVE * CAP_J)
        if revive.any():
            self.alive[revive] = True
        # listening cost
        pl = np.where(self.is_ch, P_LISTEN_CH, P_LISTEN_MEMBER)
        e = pl * DT * self.alive
        self.store -= e
        self.m['energy'] += float(e.sum())
        died = self.alive & (self.store <= 0)
        self.store[died] = 0
        self.alive[died] = False
        if t % 30 == 0:
            self._cache = {}
        # link fading
        self.fade = FADE_RHO * self.fade + np.sqrt(1 - FADE_RHO ** 2) * self.rng.normal(0, FADE_SIG, self.N + 1)
        self.fade[self.S] = 0.0
        # re-affiliate members of dead heads
        if any(not self.alive[c] for c in self.chs):
            for c in self.chs:
                if not self.alive[c] and self.buffers.get(int(c)):
                    self.m['lost_mc'] += len(self.buffers[int(c)])
                    self.buffers[int(c)] = []
            self.affiliate()
        # generation and member -> head hop
        gen = np.where((self.next_gen <= t) & self.alive)[0]
        self.next_gen[self.next_gen <= t] += cfg.gen_period
        self.m['gen'] += len(gen)
        for i in gen:
            c = self.member_of[i]
            if c < 0:
                continue
            if c == i:
                self.buffers[int(c)].append((i, float(t)))
                continue
            p = self.link_prr(i, c)
            att = 0
            ok = False
            while att < MAX_ATT and not ok:
                att += 1
                ok = self.rng.random() < p
            ft = frame_time(HDR_BYTES + PAYLOAD)
            if not self.spend(i, P_TX * (T_W_CH * 0.05 + att * ft)):
                continue
            self.spend(c, P_RX * ft)
            if ok and self.alive[c]:
                self.buffers[int(c)].append((i, float(t)))
            else:
                self.m['lost_mc'] += 1

    def account(self):
        t = self.t
        for (s, g) in self.deliv.pop(t, []):
            if g > self.U[s]:
                self.U[s] = g
        age = t - self.U
        # peak AoI: age just before an update lowers it
        drop = self.U > self.last_U
        if drop.any():
            self.m['peak'].extend(list(self.peak_track[drop]))
        self.peak_track = age + DT
        self.last_U = self.U.copy()
        self.m['aoi_sum'] += float(age.sum())
        self.m['aoi_n'] += self.N
        self.m['dead_frac'] += float((~self.alive & ~self.failed).mean())

    def summary(self):
        m = self.m
        T = max(self.t, 1)
        gen = max(m['gen'], 1)
        pk = np.array(m['peak']) if m['peak'] else np.array([self.t])
        return dict(aoi=m['aoi_sum'] / max(m['aoi_n'], 1),
                    paoi95=float(np.percentile(pk, 95)),
                    viol=1.0 - m['ontime'] / gen,
                    pdr=m['deliv'] / gen,
                    plr=1.0 - m['deliv'] / gen,
                    e_bit=1e6 * m['energy'] / max(m['bits'], 1.0),
                    energy=m['energy'], harvest=m['harvest'],
                    brownout=m['dead_frac'] / T,
                    ctrl=m['ctrl_frames'], batches=m['batches'],
                    mean_batch=m['released'] / max(m['batches'], 1),
                    shield_raise=m['shield_raise'], infeasible=m['infeasible'],
                    lost_mc=m['lost_mc'] / gen, lost_dead=m['lost_dead'], lost_hop=m['lost_hop'], gen=m['gen'], released=m['released'])
