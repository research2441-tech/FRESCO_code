"""Controllers: neuro-fuzzy elector, adaptive conformal shield, GAT-MAPPO with Lagrangian duals,
and the re-implemented baselines. All controllers run inside the same simulator (fresco_sim)."""
import math
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fresco_sim as fs

torch.set_num_threads(1)
HOLD = 0  # action 0 = hold, action 1+k = release with R_LEVELS[k] repair packets
N_FEAT = 15


def k_heads(N):
    return max(3, int(math.ceil(0.1 * N)))


# ============================================================================ elector
class NeuroFuzzyElector:
    """Zero-order Takagi-Sugeno elector with Gaussian memberships (3 terms per input)."""
    TERMS = 3
    N_IN = 4

    def __init__(self, theta=None):
        self.n_rules = self.TERMS ** self.N_IN
        if theta is None:
            theta = self.default()
        self.set(theta)
        m = np.array(np.meshgrid(*[range(self.TERMS)] * self.N_IN, indexing='ij')).reshape(self.N_IN, -1).T
        self.rule_terms = m  # (R, 4)

    @classmethod
    def dim(cls):
        return 2 * cls.N_IN * cls.TERMS + cls.TERMS ** cls.N_IN

    def default(self):
        c = np.tile([0.0, 0.5, 1.0], self.N_IN)
        s = np.full(self.N_IN * self.TERMS, 0.25)
        m = np.array(np.meshgrid(*[range(self.TERMS)] * self.N_IN, indexing='ij')).reshape(self.N_IN, -1).T
        # prior: high energy, high harvest, low ETX, low relay load
        w = (m[:, 0] + m[:, 1] + (2 - m[:, 2]) + (2 - m[:, 3])) / 8.0
        return np.concatenate([c, np.log(s), w])

    def set(self, theta):
        theta = np.asarray(theta, float)
        k = self.N_IN * self.TERMS
        self.theta = theta
        self.c = theta[:k].reshape(self.N_IN, self.TERMS)
        self.s = np.exp(np.clip(theta[k:2 * k], -4, 1)).reshape(self.N_IN, self.TERMS)
        self.w = theta[2 * k:]

    def memberships(self, X):
        return np.exp(-(X[:, :, None] - self.c[None]) ** 2 / (2 * self.s[None] ** 2))  # (n,4,3)

    def score(self, X):
        mu = self.memberships(X)
        fire = np.ones((X.shape[0], self.n_rules))
        for k in range(self.N_IN):
            fire *= mu[:, k, self.rule_terms[:, k]]
        fire_n = fire / (fire.sum(1, keepdims=True) + 1e-12)
        return fire_n @ self.w

    def __call__(self, env, K):
        X, _ = env.node_features()
        return env.elect(self.score(X), K)


def heed_elect(env, K):
    X, dsink = env.node_features()
    return env.elect(X[:, 0] + 0.05 * env.rng.random(env.N), K)


class LeachElect:
    def __init__(self, p=0.1):
        self.p = p
        self.last = {}

    def __call__(self, env, K):
        rnd = env.t // fs.ROUND
        per = int(round(1 / self.p))
        elig = [i for i in range(env.N) if env.alive[i] and (rnd - self.last.get(i, -10 ** 6)) >= per]
        if len(elig) < K:
            elig = [i for i in range(env.N) if env.alive[i]]
        ch = env.rng.choice(elig, size=min(K, len(elig)), replace=False)
        for i in ch:
            self.last[int(i)] = rnd
        return np.array(ch, int)


def fbchs_elect(env, K):
    """Region-based fuzzy head selection in the style of FBCHS (five fixed regions)."""
    L = env.L
    a, b, c, d = 0.0, 0.2 * L, 0.8 * L, L
    x, y = env.pos[:, 0], env.pos[:, 1]
    reg = np.full(env.N, 4)
    reg[(x < b)] = 0
    reg[(x >= c)] = 1
    reg[(x >= b) & (x < c) & (y < b)] = 2
    reg[(x >= b) & (x < c) & (y >= c)] = 3
    e = env.store / fs.CAP_J
    dn = np.clip(env.dist[:env.N, env.S] / (1.2 * L), 0, 1)
    mu_high = np.clip((e - 0.2) / 0.6, 0, 1)
    mu_near = np.clip(1 - dn, 0, 1)
    sc = np.minimum(mu_high, mu_near) * (env.alive & (env.store > 0.15 * fs.CAP_J))
    per = int(np.ceil(K / 5))
    out = []
    for r_ in range(5):
        idx = np.where(reg == r_)[0]
        if len(idx):
            out.extend(idx[np.argsort(-sc[idx])][:per].tolist())
    return np.array(out[:K] if len(out) >= K else out, int)


# ============================================================================ conformal shield
class ConformalShield:
    """Online calibration of the per-update delivery ratio s = T_i / d_hat(r), with one score window
    per repair level r (Mondrian taxonomy) and one adaptive working level shared by all strata.
    T_i is the delivery delay of update i measured from batch release; a lost update has s = +inf.

    mode = 'aci'    adaptive conformal inference: alpha_{t+1} = alpha_t + gamma (alpha - err)
           'static' split-conformal quantile frozen after W_CAL scores per stratum
           'gauss'  mean + z * std of the finite ratios (parametric bound)
           'none'   nominal bound (ratio 1) and no enforcement
    """
    W_CAL = 300
    N_MIN = 30

    def __init__(self, alpha, mode='aci', gamma=0.002, window=1500, enforce=True, lookahead=1):
        self.alpha = alpha
        self.mode = mode
        self.gamma = gamma
        nr = len(fs.R_LEVELS)
        self.alpha_t = alpha            # one working level shared by all strata
        self.scores = [deque(maxlen=window) for _ in range(nr)]
        self.static_q = [None] * nr
        self.enforce = enforce and mode != 'none'
        self.pending = {}
        self.err_hist = []      # one entry per released update: 1 if its score exceeded the bound used
        self.miss_hist = []     # 1 if the update missed the deadline (released updates only)
        self.miss_feas_hist = []  # misses of updates released through a feasible shield decision
        self.feasible = False
        self.lookahead = lookahead

    def _q_one(self, k):
        if self.mode == 'none':
            return 1.0
        s = self.scores[k]
        n = len(s)
        if n < self.N_MIN:
            return 4.0
        arr = np.fromiter(s, float)
        if self.mode == 'gauss':
            fin = arr[np.isfinite(arr) & (arr < 1e5)]
            if len(fin) < 5:
                return 4.0
            return float(fin.mean() + _z(self.alpha) * fin.std())
        if self.mode == 'static':
            if self.static_q[k] is None:
                lev = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
                qv = float(np.quantile(arr, lev, method='higher'))
                if n >= self.W_CAL:
                    self.static_q[k] = qv
                return qv if qv < 1e5 else np.inf
            return self.static_q[k] if self.static_q[k] < 1e5 else np.inf
        a = self.alpha_t
        if a <= 0:
            return np.inf
        if a >= 1:
            return 0.0
        lev = min(1.0, math.ceil((n + 1) * (1 - a)) / n)
        qv = float(np.quantile(arr, lev, method='higher'))
        return qv if qv < 1e5 else np.inf

    def q(self):
        if getattr(self, '_qc', None) is None:
            self._qc = np.array([self._q_one(k) for k in range(len(fs.R_LEVELS))])
        return self._qc

    def schedule(self, t_now, rec, q_used, deadline):
        """Feedback reaches the head when an update is delivered, or at the deadline when it is not."""
        if rec is None:
            return
        feas = self.feasible
        self.feasible = False
        k = fs.R_LEVELS.index(rec['r']) if rec['r'] in fs.R_LEVELS else len(fs.R_LEVELS) - 1
        qk = q_used[k] if np.ndim(q_used) else q_used
        for T_i, g in zip(rec['deliver_t'], rec['gens']):
            s = T_i / rec['d_hat']
            late = (t_now + T_i) - g > deadline
            t_fb = t_now + int(math.ceil(min(T_i, deadline)))
            self.pending.setdefault(t_fb, []).append((k, s, qk, late, feas))

    def tick(self, t):
        self._qc = None
        for k, s, qk, late, feas in self.pending.pop(t, []):
            err = float(s > qk)
            if self.mode == 'aci':
                self.alpha_t = self.alpha_t + self.gamma * (self.alpha - err)
            self.scores[k].append(s if np.isfinite(s) else 1e6)
            self.err_hist.append(err)
            self.miss_hist.append(float(late))
            if feas:
                self.miss_feas_hist.append(float(late))

    def apply(self, env, ch, action, deadline):
        """Executed action after the deadline check, and the calibrated ratios in force.
        Sets self.feasible for the release that follows (True if its bound meets the deadline)."""
        qr = self.q()
        self.feasible = False
        buf = env.buffers.get(ch, [])
        if not buf:
            return HOLD, qr, False
        if not self.enforce:
            return action, qr, False
        age = env.t - min(g for _, g in buf)
        U, _ = env.predict_bounds(ch, qr)
        if not np.isfinite(U).any():
            # no repair level has a finite calibrated bound: the deadline cannot be certified
            if action == HOLD:
                return HOLD, qr, False
            env.m['infeasible'] += 1
            top = len(fs.R_LEVELS)
            return top, qr, action != top
        feas = np.where(age + U <= deadline)[0]
        if action == HOLD:
            if age + self.lookahead * fs.DT + np.nanmin(U) > deadline:
                if len(feas):
                    k = feas[0]
                    self.feasible = True
                else:
                    k = int(np.argmin(U))
                    env.m['infeasible'] += 1
                env.m['shield_raise'] += 1
                return 1 + k, qr, True
            return HOLD, qr, False
        k = action - 1
        if age + U[k] <= deadline:
            self.feasible = True
            return action, qr, False
        up = feas[feas >= k] if len(feas) else feas
        if len(up):
            k2 = up[0]
            self.feasible = True
        elif len(feas):
            k2 = feas[-1]
            self.feasible = True
        else:
            k2 = int(np.argmin(U))
            env.m['infeasible'] += 1
        env.m['shield_raise'] += 1
        return 1 + k2, qr, True


def _z(a):
    from scipy.stats import norm
    return float(norm.ppf(1 - a))


# ============================================================================ observations
def ch_observation(env, ch, qr, deadline):
    buf = env.buffers.get(ch, [])
    D = deadline
    x = np.zeros(N_FEAT, np.float32)
    x[0] = len(buf) / fs.B_MAX
    if buf:
        g = np.array([g for _, g in buf])
        x[1] = min((env.t - g.min()) / D, 2.0)
        x[2] = min((env.t - g.mean()) / D, 2.0)
        U, dh = env.predict_bounds(ch, np.minimum(qr, 50.0))
        x[3] = np.clip((D - (env.t - g.min()) - U[0]) / D, -1, 1)
        x[4] = np.clip((D - (env.t - g.min()) - U[-1]) / D, -1, 1)
        x[5] = min(dh[0] / D, 2.0)
    paths = env.paths.get(ch, [])
    x[6] = len(paths) / 3.0
    if paths:
        qs = [env.path_stats(p)[0] for p in paths]
        x[7] = qs[0]
        x[8] = float(np.mean(qs))
        rel = [v for p in paths for v in p[1:-1]]
        x[11] = min(env.store[rel].min() / fs.CAP_J, 1.0) if rel else 1.0
    x[9] = env.store[ch] / fs.CAP_J
    x[10] = min(env.harv_ewma[ch] / (fs.PANEL * 1000.0), 1.0)
    ph = 2 * np.pi * (env.t % fs.DAY_STEPS) / fs.DAY_STEPS
    x[12], x[13] = np.sin(ph), np.cos(ph)
    x[14] = (env.member_of == ch).sum() / (env.N / max(len(env.chs), 1))
    return x


def ch_graph(env, k=3):
    P = env.pos[env.chs]
    K = len(env.chs)
    d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
    A = np.eye(K, dtype=np.float32)
    for i in range(K):
        nn_ = np.argsort(d[i])[1:k + 1]
        A[i, nn_] = 1
    return A


# ============================================================================ networks
TAUS = (2.0, 4.0, 7.0, 11.0, 16.0, 24.0)   # candidate batch-release ages (s)
N_TAU, N_R = len(TAUS), len(fs.R_LEVELS)


class GATLayer(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.W = nn.Linear(fin, fout, bias=False)
        self.a_src = nn.Parameter(torch.randn(fout) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(fout) * 0.1)

    def forward(self, x, A):  # x (B,K,F), A (B,K,K)
        h = self.W(x)
        e = (h @ self.a_dst).unsqueeze(-1) + (h @ self.a_src).unsqueeze(-2)
        e = F.leaky_relu(e, 0.2).masked_fill(A < 0.5, -1e9)
        att = torch.softmax(e, -1)
        return F.elu(att @ h), att


class Actor(nn.Module):
    """Factorised release policy: a release age tau and a repair level r per batch."""

    def __init__(self, use_gat=True, hid=64):
        super().__init__()
        self.use_gat = use_gat
        self.gat = GATLayer(N_FEAT, 32) if use_gat else None
        fin = N_FEAT + (32 if use_gat else 0)
        self.body = nn.Sequential(nn.Linear(fin, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
        self.h_tau = nn.Linear(hid, N_TAU)
        self.h_r = nn.Linear(hid, N_R)
        with torch.no_grad():
            for h in (self.h_tau, self.h_r):
                h.weight.mul_(0.1)
                h.bias.zero_()

    def forward(self, x, A):
        if self.use_gat:
            z, _ = self.gat(x, A)
            x = torch.cat([x, z], -1)
        h = self.body(x)
        return self.h_tau(h), self.h_r(h)


class Critic(nn.Module):
    """Centralised critic with separate heads for return and constraint cost."""

    def __init__(self, hid=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(2 * N_FEAT, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh())
        self.v = nn.Linear(hid, 3)   # return, energy-budget cost, reserve cost

    def forward(self, x):
        g = x.mean(-2, keepdim=True).expand_as(x)
        h = self.body(torch.cat([x, g], -1))
        return self.v(h)


# ============================================================================ episode runner
def reserve_cost(env, ch):
    rel = {v for p in env.paths.get(ch, []) for v in p[1:-1]}
    nodes = [ch] + sorted(rel)
    st = env.store[nodes] / fs.CAP_J
    al = env.alive[nodes]
    return float(np.mean((st < 0.1) | (~al)))


def member_aoi(env, ch, D):
    mem = np.where(env.member_of == ch)[0]
    if len(mem) == 0:
        return 0.0
    return float((env.t - env.U[mem]).mean() / D)


class Controller:
    """Base controller: elector + release rule + optional shield; coded flag for RLNC."""
    name = 'base'
    coded = True
    k_paths = 3
    learner = None

    def __init__(self, elector, shield_mode='none', alpha=0.05):
        self.elector = elector
        self.shield_mode = shield_mode
        self.alpha = alpha

    def reset(self, env):
        self.shield = ConformalShield(self.alpha, self.shield_mode)

    def act(self, env, obs_ctx):
        raise NotImplementedError

    def released(self, ch):
        pass

    def flush_r(self, ch):
        return 1


def run_episode(cfg, seed, ctrl, learner=None, record=False):
    cfg = fs.Config(cfg)
    cfg['coding'] = ctrl.coded
    cfg['k_paths'] = ctrl.k_paths
    env = fs.WSN(cfg, seed)
    ctrl.learner = learner
    ctrl.reset(env)
    ctrl.log_dec = [] if record else None
    K = k_heads(env.N)
    D = cfg.deadline
    trace = dict(t=[], store_ch=[], harvest=[], alpha=[], viol_roll=[], q=[], ghi=[]) if record else None
    A = None
    if learner is not None:
        learner.begin(ctrl, K)
    for t in range(cfg.steps):
        env.t = t
        if t % fs.ROUND == 0:
            # outgoing heads hand over by flushing their buffers before the new election
            for ch in list(env.buffers):
                if env.buffers[ch] and env.alive[ch]:
                    r0 = ctrl.flush_r(ch)
                    a_f = 1 + (fs.R_LEVELS.index(r0) if r0 in fs.R_LEVELS else 0)
                    a_f, q_used, _ = ctrl.shield.apply(env, ch, a_f, D)
                    r = ctrl.repairs(env, ch, a_f) if hasattr(ctrl, 'repairs') else fs.R_LEVELS[a_f - 1]
                    rec = env.release(ch, r, coded=ctrl.coded)
                    ctrl.shield.schedule(t, rec, q_used, D)
                    ctrl.released(ch)
            chs = ctrl.elector(env, K)
            env.set_heads(chs)
            A = ch_graph(env)
            if hasattr(ctrl, 'plan'):
                ctrl.plan = {}
        env.step_physics()
        ctrl.shield.tick(t)
        qr = ctrl.shield.q()
        acts, _ = ctrl.act(env, dict(qr=qr, A=A, D=D))
        for i, ch in enumerate(env.chs):
            ch = int(ch)
            if not env.alive[ch]:
                continue
            a = acts[i]
            buf = env.buffers.get(ch, [])
            if len(buf) >= fs.B_MAX and a == HOLD:
                a = 1 + ctrl.flush_r(ch)
            a_exec, q_used, over = ctrl.shield.apply(env, ch, a, D)
            if a_exec != HOLD and env.buffers.get(ch):
                r = ctrl.repairs(env, ch, a_exec) if hasattr(ctrl, 'repairs') else fs.R_LEVELS[a_exec - 1]
                rec = env.release(ch, r, coded=ctrl.coded)
                ctrl.shield.schedule(t, rec, q_used, D)
                ctrl.released(ch)
                if learner is not None and rec is not None:
                    learner.on_release(i, rec['energy'])
            if learner is not None and over:
                learner.store_override(i)
        env.account()
        if learner is not None:
            learner.on_step(env, D)
        if record and t % 10 == 0:
            trace['t'].append(t)
            trace['store_ch'].append(float(env.store[env.chs].mean()) / fs.CAP_J)
            trace['ghi'].append(float(env.ghi_t[t]))
            trace['alpha'].append(float(ctrl.shield.alpha_t))
            mh = ctrl.shield.miss_hist[-300:]
            eh = ctrl.shield.err_hist[-300:]
            trace['viol_roll'].append(float(np.mean(mh)) if mh else np.nan)
            trace['q'].append(float(np.mean(eh)) if eh else np.nan)
    if learner is not None:
        learner.end()
    env.t = cfg.steps
    out = env.summary()
    eh = ctrl.shield.err_hist
    out['miscover'] = float(np.mean(eh)) if eh else np.nan
    out['miss_released'] = float(np.mean(ctrl.shield.miss_hist)) if ctrl.shield.miss_hist else np.nan
    mf = ctrl.shield.miss_feas_hist
    out['miss_feasible'] = float(np.mean(mf)) if mf else np.nan
    out['feasible_frac'] = len(mf) / max(len(ctrl.shield.miss_hist), 1)
    out['miss_feas_share'] = float(np.sum(mf)) / max(len(ctrl.shield.miss_hist), 1)
    if record:
        trace['decisions'] = ctrl.log_dec
        out['trace'] = trace
    return out


# ============================================================================ rule-based controllers
class Immediate(Controller):
    name = 'LEACH-Direct'
    coded = False
    k_paths = 1

    def act(self, env, ctx):
        return [1 if env.buffers.get(int(c)) else HOLD for c in env.chs], None

    def flush_r(self, ch):
        return 0


class MMSPEEDLike(Controller):
    """Immediate forwarding with probabilistic multipath duplication to reach rho."""
    name = 'HEED-MMSPEED'
    coded = False
    rho = 0.95

    def act(self, env, ctx):
        return [1 if env.buffers.get(int(c)) else HOLD for c in env.chs], None

    def repairs(self, env, ch, a):
        paths = env.paths.get(ch, [])
        if not paths:
            return 0
        qs = sorted([env.path_stats(p)[0] for p in paths], reverse=True)
        fail = 1.0
        for k, q in enumerate(qs):
            fail *= (1 - q)
            if 1 - fail >= self.rho:
                return k * len(env.buffers[ch])
        return (len(qs) - 1) * len(env.buffers[ch])

    def flush_r(self, ch):
        return 0


class FBMPBatch(Controller):
    """FBCHS heads with fixed-size or timeout batches duplicated on two disjoint paths."""
    name = 'FB-MPBatch'
    coded = False
    k_paths = 2
    batch = 8
    timeout = 20.0

    def act(self, env, ctx):
        out = []
        for c in env.chs:
            buf = env.buffers.get(int(c), [])
            rel = buf and (len(buf) >= self.batch or env.t - min(g for _, g in buf) >= self.timeout)
            out.append(1 if rel else HOLD)
        return out, None

    def repairs(self, env, ch, a):
        return len(env.buffers[ch]) if len(env.paths.get(ch, [])) > 1 else 0

    def flush_r(self, ch):
        return 0


class StaticCoded(Controller):
    name = 'SCM-RLNC'
    timeout = 10.0
    r = 2

    def act(self, env, ctx):
        out = []
        for c in env.chs:
            buf = env.buffers.get(int(c), [])
            rel = buf and env.t - min(g for _, g in buf) >= self.timeout
            out.append(1 + self.r if rel else HOLD)
        return out, None

    def flush_r(self, ch):
        return self.r


class AgeThreshold(Controller):
    """Energy-adaptive age-threshold release, single path, uncoded (MAoIG-inspired)."""
    name = 'EA-AgeThr'
    coded = False
    k_paths = 1
    tau0 = 10.0

    def act(self, env, ctx):
        out = []
        for c in env.chs:
            buf = env.buffers.get(int(c), [])
            if not buf:
                out.append(HOLD)
                continue
            e = env.store[int(c)] / fs.CAP_J
            tau = self.tau0 * (1 + (1 - e))
            out.append(1 if env.t - min(g for _, g in buf) >= tau else HOLD)
        return out, None

    def flush_r(self, ch):
        return 0


class DriftPlusPenalty(Controller):
    """Lyapunov drift-plus-penalty with energy-deficit and deadline virtual queues."""
    name = 'Lyapunov-DPP'
    V = 1.0

    def reset(self, env):
        super().reset(env)
        self.QE = {}
        self.QD = 0.0

    def act(self, env, ctx):
        D = ctx['D']
        out = []
        ph = fs.PANEL * env.ghi_t[env.t] * env.cfg.harvest_scale
        for c in env.chs:
            c = int(c)
            buf = env.buffers.get(c, [])
            qe = self.QE.get(c, 0.0)
            pc = fs.P_LISTEN_CH - ph * env.shade[c]
            self.QE[c] = max(qe + pc * fs.DT / 0.01, 0.0)
            if not buf:
                out.append(HOLD)
                continue
            g = np.array([g for _, g in buf])
            age = env.t - g.min()
            U, dh = env.predict_bounds(c, 1.0)
            best, ba = self.V * len(buf) * fs.DT / D, HOLD  # cost of holding: aging of the buffer
            hops = np.mean([len(p) - 1 for p in env.paths.get(c, [[0, 0]])])
            for k, r in enumerate(fs.R_LEVELS):
                e_cost = (hops * fs.STROBE * fs.P_TX * (1 + (r > 0))) / 0.01
                miss = float(age + dh[k] * 1.5 > D)
                cost = (self.QE[c] / 100.0) * e_cost + self.QD * miss + 0.02 * r
                if cost < best:
                    best, ba = cost, 1 + k
            out.append(ba)
            self.QD = max(self.QD + (float(age + fs.DT + dh.min() > D) - 0.05) * 0.1, 0.0)
        return out, None


# ============================================================================ learned (semi-Markov) release
class PlanController(Controller):
    """When a head's buffer opens a new batch, the agent fixes a release age tau and a repair
    level r; the batch is released once its oldest update reaches tau (or the shield intervenes)."""

    def reset(self, env):
        super().reset(env)
        self.plan = {}

    def decide(self, X, A, idx):
        raise NotImplementedError

    def act(self, env, ctx):
        chs = [int(c) for c in env.chs]
        need = [i for i, c in enumerate(chs) if env.alive[c] and env.buffers.get(c) and c not in self.plan]
        if need:
            X = np.stack([ch_observation(env, c, ctx['qr'], ctx['D']) for c in chs])
            dec = self.decide(X, ctx['A'], need)
            for i, (ti, ri, lp) in zip(need, dec):
                self.plan[chs[i]] = (ti, ri)
                if getattr(self, 'log_dec', None) is not None:
                    self.log_dec.append((int(env.t), int(ti), int(ri), float(X[i, 9]), float(X[i, 10]),
                                         float(X[i, 7])))
                if self.learner is not None:
                    self.learner.on_decision(i, X, ctx['A'], ti, ri, lp)
        acts = []
        for c in chs:
            buf = env.buffers.get(c, [])
            if not buf or c not in self.plan:
                acts.append(HOLD)
                continue
            ti, ri = self.plan[c]
            age = env.t - min(g for _, g in buf)
            acts.append(1 + ri if age >= TAUS[ti] else HOLD)
        return acts, None

    def released(self, ch):
        self.plan.pop(ch, None)

    def flush_r(self, ch):
        p = self.plan.get(ch)
        return fs.R_LEVELS[p[1]] if p else 1


class MAPPOController(PlanController):
    name = 'FRESCO'

    def __init__(self, elector, actor, shield_mode='aci', alpha=0.05, greedy=True):
        super().__init__(elector, shield_mode, alpha)
        self.actor = actor
        self.greedy = greedy

    def decide(self, X, A, idx):
        with torch.no_grad():
            lt, lr = self.actor(torch.from_numpy(X)[None], torch.from_numpy(A)[None])
        lt, lr = lt[0, idx], lr[0, idx]
        if self.greedy:
            return [(int(a), int(b), 0.0) for a, b in zip(lt.argmax(-1), lr.argmax(-1))]
        dt, dr = torch.distributions.Categorical(logits=lt), torch.distributions.Categorical(logits=lr)
        at, ar = dt.sample(), dr.sample()
        lp = (dt.log_prob(at) + dr.log_prob(ar)).numpy()
        return [(int(a), int(b), float(l)) for a, b, l in zip(at, ar, lp)]


class SMDPBuffer:
    """Per-agent-slot accumulation of discounted reward/cost between consecutive decisions."""

    def __init__(self, gamma, override_pen, pen_fixed=0.0):
        self.gamma = gamma
        self.override_pen = override_pen

    def begin(self, K):
        self.K = K
        self.open = [None] * K
        self.trans = [[] for _ in range(K)]

    def on_decision(self, i, X, A, ti, ri, lp):
        if self.open[i] is not None:
            o = self.open[i]
            o['next'] = (X, A)
            self.trans[i].append(o)
        self.open[i] = dict(X=X, A=A, ti=ti, ri=ri, lp=lp, R=0.0, C=0.0, E=0.0, k=0, next=None)

    def add(self, i, r, c):
        o = self.open[i]
        if o is None:
            return
        g = self.gamma ** o['k']
        o['R'] += g * r
        o['C'] += g * c
        o['k'] += 1

    def add_energy(self, i, e):
        o = self.open[i]
        if o is not None:
            o['E'] += (self.gamma ** o['k']) * e

    def override(self, i):
        if self.open[i] is not None:
            self.open[i]['R'] -= self.override_pen * (self.gamma ** self.open[i]['k'])

    def end(self):
        for i in range(self.K):
            if self.open[i] is not None:
                self.trans[i].append(self.open[i])
                self.open[i] = None


P_REF = 0.01   # 10 mW reference power for the energy-budget cost


class MAPPOLearner:
    """Lagrangian MAPPO on the semi-Markov batch decisions of all heads.

    Two constraints: the communication energy rate of each cluster (normalised by P_REF) must not
    exceed eps_e, and the share of the head and its relays that sit below the 10 % storage reserve
    must not exceed eps_r.  Duals follow projected ascent on the relative violation."""

    def __init__(self, actor, critic, eps_e=0.3, eps_r=0.02, lagrangian=True, fixed_pen=1.0, lr=3e-4,
                 gamma=0.99, lam=0.95, clip=0.2, epochs=6, ent=0.01, lr_dual=0.1, override_pen=0.05):
        self.actor, self.critic = actor, critic
        self.opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(critic.parameters(), lr=lr)
        self.eps = np.array([eps_e, eps_r])
        self.lagrangian = lagrangian
        self.lmb = np.array([0.5, 0.5]) if lagrangian else np.array([fixed_pen, fixed_pen])
        self.gamma, self.lam, self.clip, self.epochs, self.ent = gamma, lam, clip, epochs, ent
        self.lr_dual = lr_dual
        self.buf = SMDPBuffer(gamma, override_pen)
        self.hist = []

    def begin(self, ctrl, K):
        self.ctrl = ctrl
        self.buf.begin(K)
        self.step_cost = []
        self.e_total = 0.0
        self.n_slot_steps = 0

    def on_decision(self, i, X, A, ti, ri, lp):
        self.buf.on_decision(i, X, A, ti, ri, lp)

    def on_release(self, i, e):
        self.buf.add_energy(i, e / (P_REF * fs.DT))
        self.e_total += e / (P_REF * fs.DT)

    def store_override(self, i):
        self.buf.override(i)

    def on_step(self, env, D):
        cs = []
        for i, c in enumerate(env.chs):
            r = -member_aoi(env, int(c), D)
            cc = reserve_cost(env, int(c))
            cs.append(cc)
            self.buf.add(i, r, cc)
        self.step_cost.append(float(np.mean(cs)))
        self.n_slot_steps += len(env.chs)

    def end(self):
        self.buf.end()

    def _gae(self, R, v, vn, gk, has_next):
        adv = np.zeros(len(R))
        pos = 0
        for seq in self.buf.trans:
            n = len(seq)
            last = 0.0
            for j in reversed(range(pos, pos + n)):
                d = R[j] + gk[j] * vn[j] - v[j]
                last = d + gk[j] * self.lam * last * has_next[j]
                adv[j] = last
            pos += n
        return adv

    def update(self):
        rows = [(i, o) for i, seq in enumerate(self.buf.trans) for o in seq]
        if not rows:
            return dict(ret=0, cost_e=0, cost_r=0, lmb_e=self.lmb[0], lmb_r=self.lmb[1])
        X = torch.from_numpy(np.stack([o['X'] for _, o in rows]))
        A = torch.from_numpy(np.stack([o['A'] for _, o in rows]))
        idx = torch.tensor([i for i, _ in rows])
        has_next = np.array([o['next'] is not None for _, o in rows], float)
        Xn = torch.from_numpy(np.stack([o['next'][0] if o['next'] is not None else o['X'] for _, o in rows]))
        ar = torch.arange(len(rows))
        with torch.no_grad():
            v = self.critic(X)[ar, idx].numpy()      # (n,3)
            vn = self.critic(Xn)[ar, idx].numpy() * has_next[:, None]
        R = np.array([o['R'] for _, o in rows])
        E = np.array([o['E'] for _, o in rows])
        C = np.array([o['C'] for _, o in rows])
        gk = self.gamma ** np.array([o['k'] for _, o in rows])
        adv = [self._gae(Z, v[:, k], vn[:, k], gk, has_next) for k, Z in enumerate((R, E, C))]
        rets = [adv[k] + v[:, k] for k in range(3)]
        j_e = self.e_total / max(self.n_slot_steps, 1)
        j_r = float(np.mean(self.step_cost))
        if self.lagrangian:
            self.lmb = np.maximum(0.0, self.lmb + self.lr_dual * np.clip(np.array([j_e, j_r]) / self.eps - 1.0, -1.0, 1.0))
        a = adv[0] - self.lmb[0] * adv[1] - self.lmb[1] * adv[2]
        a = (a - a.mean()) / (a.std() + 1e-8)
        adv_t = torch.from_numpy(a).float()
        ret_t = torch.from_numpy(np.stack(rets, 1)).float()
        ti = torch.tensor([o['ti'] for _, o in rows])
        ri = torch.tensor([o['ri'] for _, o in rows])
        lp_old = torch.tensor([o['lp'] for _, o in rows]).float()
        T = len(rows)
        mb = 512
        for _ in range(self.epochs):
            perm = torch.randperm(T)
            for s in range(0, T, mb):
                b = perm[s:s + mb]
                bb = torch.arange(len(b))
                lt, lr = self.actor(X[b], A[b])
                dt = torch.distributions.Categorical(logits=lt[bb, idx[b]])
                dr = torch.distributions.Categorical(logits=lr[bb, idx[b]])
                lp = dt.log_prob(ti[b]) + dr.log_prob(ri[b])
                ratio = torch.exp(lp - lp_old[b])
                a_ = adv_t[b]
                la = -torch.min(ratio * a_, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a_).mean()
                la = la - self.ent * (dt.entropy() + dr.entropy()).mean()
                self.opt_a.zero_grad()
                la.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.opt_a.step()
                pv = self.critic(X[b])[bb, idx[b]]
                lc = F.mse_loss(pv, ret_t[b])
                self.opt_c.zero_grad()
                lc.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.opt_c.step()
        info = dict(ret=float(R.mean()), cost_e=j_e, cost_r=j_r, lmb_e=float(self.lmb[0]),
                    lmb_r=float(self.lmb[1]), n=T)
        self.hist.append(info)
        return info


def train_mappo(cfg, elector, seeds, use_gat=True, lagrangian=True, coded=True, shield_mode='aci',
                episodes=60, eps_e=0.3, eps_r=0.02, fixed_pen=1.0, torch_seed=0, log=None):
    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    actor, critic = Actor(use_gat), Critic()
    learner = MAPPOLearner(actor, critic, eps_e=eps_e, eps_r=eps_r, lagrangian=lagrangian, fixed_pen=fixed_pen)
    ctrl = MAPPOController(elector, actor, shield_mode=shield_mode, alpha=cfg['alpha'], greedy=False)
    ctrl.coded = coded
    for ep in range(episodes):
        t0 = time.time()
        res = run_episode(cfg, int(seeds[ep % len(seeds)]), ctrl, learner=learner)
        info = learner.update()
        info.update(ep=ep, aoi=res['aoi'], viol=res['viol'], brown=res['brownout'], ebit=res['e_bit'],
                    sec=time.time() - t0)
        if log is not None:
            log.append(info)
    return actor, critic, learner


# ============================================================================ independent DQN (SMDP)
class QNet(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(N_FEAT, hid), nn.ReLU(), nn.Linear(hid, hid), nn.ReLU(),
                               nn.Linear(hid, N_TAU * N_R))

    def forward(self, x):
        return self.f(x)


class DQNController(PlanController):
    name = 'IQL-DQN'

    def __init__(self, elector, qnet, eps=0.0):
        super().__init__(elector, 'none')
        self.q = qnet
        self.eps = eps
        self.rng = np.random.default_rng(0)

    def decide(self, X, A, idx):
        with torch.no_grad():
            a = self.q(torch.from_numpy(X[idx])).argmax(-1).numpy()
        if self.eps > 0:
            m = self.rng.random(len(a)) < self.eps
            a[m] = self.rng.integers(0, N_TAU * N_R, m.sum())
        return [(int(x // N_R), int(x % N_R), 0.0) for x in a]


class DQNLearner:
    def __init__(self, qnet, pen=1.0, gamma=0.99, lr=5e-4):
        self.q = qnet
        self.tq = QNet()
        self.tq.load_state_dict(qnet.state_dict())
        self.opt = torch.optim.Adam(qnet.parameters(), lr=lr)
        self.rb = deque(maxlen=100000)
        self.pen, self.gamma = pen, gamma
        self.buf = SMDPBuffer(gamma, 0.0)
        self.n_upd = 0

    def begin(self, ctrl, K):
        self.buf.begin(K)

    def on_decision(self, i, X, A, ti, ri, lp):
        o = self.buf.open[i]
        self.buf.on_decision(i, X, A, ti, ri, lp)
        if o is not None:
            self.rb.append((o['X'][i], o['ti'] * N_R + o['ri'], o['R'] - self.pen * (o['C'] + o['E']), o['k'], X[i], 1.0))
            self._train()

    def store_override(self, i):
        pass

    def on_release(self, i, e):
        self.buf.add_energy(i, e / (P_REF * fs.DT))

    def on_step(self, env, D):
        for i, c in enumerate(env.chs):
            self.buf.add(i, -member_aoi(env, int(c), D), reserve_cost(env, int(c)))

    def end(self):
        for i in range(self.buf.K):
            o = self.buf.open[i]
            if o is not None:
                self.rb.append((o['X'][i], o['ti'] * N_R + o['ri'], o['R'] - self.pen * (o['C'] + o['E']), o['k'], o['X'][i], 0.0))
        self.buf.end()

    def _train(self):
        if len(self.rb) < 1000:
            return
        b = [self.rb[k] for k in np.random.randint(0, len(self.rb), 128)]
        x = torch.tensor(np.stack([z[0] for z in b]))
        a = torch.tensor([z[1] for z in b]).long()
        r_ = torch.tensor([z[2] for z in b]).float()
        k = torch.tensor([z[3] for z in b]).float()
        x2 = torch.tensor(np.stack([z[4] for z in b]))
        nd = torch.tensor([z[5] for z in b]).float()
        with torch.no_grad():
            a2 = self.q(x2).argmax(-1, keepdim=True)
            y = r_ + nd * (self.gamma ** k) * self.tq(x2).gather(1, a2).squeeze(1)
        qv = self.q(x).gather(1, a[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(qv, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.n_upd += 1
        if self.n_upd % 500 == 0:
            self.tq.load_state_dict(self.q.state_dict())


def train_dqn(cfg, elector, seeds, episodes=40, pen=1.0, torch_seed=0, log=None):
    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    q = QNet()
    learner = DQNLearner(q, pen=pen)
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        ctrl = DQNController(elector, q, eps=eps)
        res = run_episode(cfg, int(seeds[ep % len(seeds)]), ctrl, learner=learner)
        if log is not None:
            log.append(dict(ep=ep, aoi=res['aoi'], viol=res['viol'], brown=res['brownout']))
    return q


# ============================================================================ CEM for the elector
class FixedReleaseShielded(Controller):
    """Release rule used while the elector is tuned: timeout plus one repair packet, ACI shield."""
    name = 'fixed'
    timeout = 12.0

    def act(self, env, ctx):
        out = []
        for c in env.chs:
            buf = env.buffers.get(int(c), [])
            rel = buf and env.t - min(g for _, g in buf) >= self.timeout
            out.append(2 if rel else HOLD)
        return out, None

    def flush_r(self, ch):
        return 1


def elector_fitness(res, D):
    return -(res['aoi'] / D + 3.0 * res['viol'] + 10.0 * res['brownout'])


def train_elector_cem(cfg, seeds, iters=8, pop=16, elite=4, rng_seed=0, log=None, pool=None):
    rng = np.random.default_rng(rng_seed)
    base = NeuroFuzzyElector()
    mu = base.default().copy()
    sd = np.concatenate([np.full(12, 0.15), np.full(12, 0.3), np.full(81, 0.3)])
    best = (-np.inf, mu.copy())
    for it in range(iters):
        cand = mu + sd * rng.standard_normal((pop, len(mu)))
        cand[0] = mu
        sds = [int(seeds[(it * 2 + k) % len(seeds)]) for k in range(2)]
        jobs = [(cfg, c, sds) for c in cand]
        fit = np.array(pool.map(_cem_eval, jobs) if pool else [_cem_eval(j) for j in jobs])
        order = np.argsort(-fit)
        el = cand[order[:elite]]
        mu = 0.7 * el.mean(0) + 0.3 * mu
        sd = 0.7 * el.std(0) + 0.3 * sd * 0.8 + 0.01
        if fit[order[0]] > best[0]:
            best = (fit[order[0]], cand[order[0]].copy())
        if log is not None:
            log.append(dict(it=it, best=float(fit[order[0]]), mean=float(fit.mean())))
    return NeuroFuzzyElector(mu), NeuroFuzzyElector(best[1])


def _cem_eval(job):
    cfg, theta, sds = job
    el = NeuroFuzzyElector(theta)
    f = 0.0
    for s in sds:
        res = run_episode(cfg, s, FixedReleaseShielded(el, 'aci', cfg['alpha']))
        f += elector_fitness(res, cfg['deadline'])
    return f / len(sds)
