"""Grayscale result figures for the manuscript (300 dpi, column or page width)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fresco_agents as fa

OUT = os.environ.get('FRESCO_OUT', '.')
FIG = os.environ.get('FRESCO_FIG', '.')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.5,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                     'axes.labelsize': 7.5, 'legend.fontsize': 6.5, 'mathtext.fontset': 'dejavuserif',
                     'lines.linewidth': 1.0, 'legend.frameon': False})
A = json.load(open(f'{OUT}/analysis.json')) if os.path.exists(f'{OUT}/analysis.json') else {}
MARK = {'FRESCO': ('*', 'black', 9), 'LEACH-Direct': ('v', '0.55', 5), 'HEED-MMSPEED': ('^', '0.55', 5),
        'FB-MPBatch': ('s', '0.35', 5), 'SCM-RLNC': ('D', '0.35', 4.5), 'EA-AgeThr': ('o', '0.2', 5),
        'Lyapunov-DPP': ('P', '0.45', 5.5), 'IQL-DQN': ('X', '0.6', 5.5), 'MAPPO-Penalty': ('h', '0.6', 5.5)}
LS = {'FRESCO': '-', 'FB-MPBatch': '--', 'SCM-RLNC': '-.', 'EA-AgeThr': ':', 'Lyapunov-DPP': (0, (5, 1, 1, 1, 1, 1))}


def save(fig, name):
    fig.savefig(f'{FIG}/{name}', dpi=300)
    plt.close(fig)


def fig_main():
    S = A['main']
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.75), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    for m, (mk, c, ms) in MARK.items():
        if m not in S:
            continue
        x, xl, xh = S[m]['e_bit']
        y, yl, yh = S[m]['aoi']
        ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]], fmt=mk, color=c, mec='black',
                    mew=0.5, ms=ms, elinewidth=0.6, capsize=1.5, label=m, zorder=3 if m == 'FRESCO' else 2)
    ax.set_xlabel('Energy per delivered bit ($\\mu$J/bit)')
    ax.set_ylabel('Average age of information (s)')
    ax.grid(True, lw=0.3, color='0.85')
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), handletextpad=0.3, borderaxespad=0.2)
    ax.text(0.02, 0.97, '(a)', transform=ax.transAxes, va='top', fontweight='bold')
    ax = axs[1]
    order = [m for m in MARK if m in S]
    xs = np.arange(len(order))
    v = [S[m]['viol'][0] * 100 for m in order]
    lo = [100 * (S[m]['viol'][0] - S[m]['viol'][1]) for m in order]
    hi = [100 * (S[m]['viol'][2] - S[m]['viol'][0]) for m in order]
    ms_ = [S[m]['miss_released'][0] * 100 for m in order]
    ax.bar(xs - 0.2, v, 0.38, yerr=[lo, hi], color=['black' if m == 'FRESCO' else '0.65' for m in order],
           ec='black', lw=0.4, error_kw=dict(lw=0.6, capsize=1.5), label='all generated updates')
    ax.bar(xs + 0.2, ms_, 0.38, color='white', ec='black', lw=0.5, hatch='////', label='released updates')
    ax.axhline(5, color='black', ls='--', lw=0.7, label='target $\\alpha=5\\%$')
    ax.set_xticks(xs)
    ax.set_xticklabels(order, rotation=40, ha='right', fontsize=6.3)
    ax.set_ylabel('Deadline-miss rate (%)')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    ax.set_ylim(0, max(v) * 1.35)
    ax.text(0.98, 0.97, '(b)', transform=ax.transAxes, va='top', ha='right', fontweight='bold')
    fig.tight_layout(w_pad=0.8)
    save(fig, 'fig3_main.png')


def fig_cd():
    F = A['friedman_aoi']
    mr = F['mean_rank']
    cd = F['cd']
    items = sorted(mr.items(), key=lambda kv: kv[1])
    k = len(items)
    fig, ax = plt.subplots(figsize=(3.5, 1.75))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0.12, 1.0)
    ax.axis('off')
    y0 = 0.80
    ax.plot([lo, hi], [y0, y0], 'k-', lw=0.8)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [y0, y0 + 0.04], 'k-', lw=0.7)
        ax.text(t, y0 + 0.07, str(t), ha='center', fontsize=6.5)
    ax.plot([lo, lo + cd], [0.95, 0.95], 'k-', lw=1.2)
    ax.text(lo + cd + 0.15, 0.95, f'CD = {cd:.2f}', ha='left', va='center', fontsize=6.3)
    half = (k + 1) // 2
    for i, (m, r) in enumerate(items):
        if i < half:
            yy = y0 - 0.12 - 0.11 * i
            ax.plot([r, r, lo - 0.05], [y0, yy, yy], 'k-', lw=0.6)
            ax.text(lo - 0.1, yy, f'{m} ({r:.2f})', ha='right', va='center', fontsize=6.2)
        else:
            j = i - half
            yy = y0 - 0.12 - 0.11 * (k - 1 - i)
            ax.plot([r, r, hi + 0.05], [y0, yy, yy], 'k-', lw=0.6)
            ax.text(hi + 0.1, yy, f'({r:.2f}) {m}', ha='left', va='center', fontsize=6.2)
    # cliques: groups whose rank span < cd
    ranks = [r for _, r in items]
    cl = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cl):
            cl.append((i, j))
    for n_, (i, j) in enumerate(cl):
        yy = y0 - 0.06 - 0.035 * n_
        ax.plot([ranks[i] - 0.03, ranks[j] + 0.03], [yy, yy], 'k-', lw=2.0)
    fig.subplots_adjust(left=0.30, right=0.70, top=0.98, bottom=0.02)
    save(fig, 'fig4_cd.png')


def fig_calibration():
    sw = A['sweep_alpha']
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
    ax = axs[0]
    marks = {'FRESCO': ('o', '-', 'ACI (FRESCO)'), 'Static split CP': ('s', '--', 'static split CP'),
             'Gaussian bound': ('^', ':', 'Gaussian bound')}
    for m, (mk, ls, lab) in marks.items():
        if m not in sw:
            continue
        al = sorted(sw[m], key=float)
        x = [float(a) for a in al]
        y = [sw[m][a]['miscover'][0] for a in al]
        yl = [sw[m][a]['miscover'][0] - sw[m][a]['miscover'][1] for a in al]
        yh = [sw[m][a]['miscover'][2] - sw[m][a]['miscover'][0] for a in al]
        ax.errorbar(x, y, yerr=[yl, yh], marker=mk, ls=ls, color='black', mfc='white' if m != 'FRESCO' else 'black',
                    ms=3.5, capsize=1.5, elinewidth=0.6, label=lab)
    xx = np.array([0.008, 0.25])
    ax.plot(xx, xx, color='0.6', lw=0.8, label='$y=\\alpha$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Target level $\\alpha$')
    ax.set_ylabel('Observed miscoverage')
    ax.grid(True, which='both', lw=0.3, color='0.88')
    ax.legend(loc='lower right', fontsize=6.0)
    ax.set_title('(a)', loc='left', fontsize=7.5, fontweight='bold')
    # (b) age versus overall miss rate: FRESCO across alpha, baselines at their tuned points
    ax = axs[1]
    f = sw['FRESCO']
    al = sorted(f, key=float)
    ax.plot([100 * f[a]['viol'][0] for a in al], [f[a]['aoi'][0] for a in al], 'k-o', ms=3.5,
            label='FRESCO, $\\alpha$ sweep', zorder=3)
    for a in al:
        off = {0.01: ((-5, 0), 'right', 'center'), 0.02: ((-5, 0), 'right', 'center'),
               0.05: ((-5, 0), 'right', 'center'), 0.1: ((0, -6), 'center', 'top'), 0.2: ((5, 0), 'left', 'center')}
        xy, ha, va = off[round(float(a), 2)]
        ax.annotate(f'$\\alpha$={float(a):g}', (100 * f[a]['viol'][0], f[a]['aoi'][0]), textcoords='offset points',
                    xytext=xy, ha=ha, va=va, fontsize=5.6)
    S = A['main']
    for m, (mk, c, ms) in MARK.items():
        if m in ('FRESCO', 'LEACH-Direct', 'HEED-MMSPEED', 'FB-MPBatch') or m not in S:
            continue
        ax.plot(100 * S[m]['viol'][0], S[m]['aoi'][0], mk, color=c, mec='black', mew=0.5, ms=ms * 0.8, ls='none',
                label=m)
    ax.set_xlabel('Deadline-miss rate, all updates (%)')
    ax.set_ylabel('Average age (s)')
    ax.grid(True, lw=0.3, color='0.88')
    ax.set_xlim(4.0, 12.5)
    ax.set_ylim(22.5, 29.8)
    ax.legend(loc='upper right', fontsize=5.6, handletextpad=0.2, labelspacing=0.25)
    ax.set_title('(b)', loc='left', fontsize=7.5, fontweight='bold')
    ax = axs[2]
    tr = json.load(open(f'{OUT}/eval_trace.json'))
    styles = {'FRESCO': ('-', 'black', 'ACI (FRESCO)'), 'Static split CP': ('--', '0.35', 'static split CP'),
              'Gaussian bound': (':', 'black', 'Gaussian bound')}
    for r in tr:
        if r.get('cfg_shift_step') is None or r['method'] not in styles:
            continue
        ls, c, lab = styles[r['method']]
        t = np.array(r['trace']['t']) / 3600 * 24
        v = np.array(r['trace']['q'], float) * 100
        ax.plot(t, v, ls=ls, color=c, lw=1.0, label=lab)
    ax.axvline(12, color='0.4', lw=0.7)
    ax.text(11.6, 0.55, '+6 dB loss on\nevery link from\n12 h onwards', transform=ax.get_xaxis_transform(),
            va='center', ha='right', fontsize=6.0)
    ax.axhline(5, color='0.6', lw=0.8, ls='-')
    ax.set_xlabel('Time of compressed day (h)')
    ax.set_ylabel('Rolling miscoverage (%)')
    ax.set_xlim(0, 24)
    ax.grid(True, lw=0.3, color='0.88')
    ax.set_ylim(-3, 112)
    ax.legend(loc='upper left', fontsize=5.6, ncol=1, handlelength=1.6)
    ax.set_title('(c)', loc='left', fontsize=7.5, fontweight='bold')
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig5_calibration.png')


def _sweep_panel(ax, key, param_vals, metric, xlabel, ylabel, scale=1.0, base=None, xlog=False):
    sw = A[key]
    main = A['main']
    for m, ls in LS.items():
        xs, ys, lo, hi = [], [], [], []
        for v in param_vals:
            if base is not None and v == base:
                st = main.get(m, {}).get(metric)
            else:
                st = sw.get(m, {}).get(str(v))
                st = st[metric] if st else None
            if st is None:
                continue
            xs.append(v)
            ys.append(st[0] * scale)
            lo.append((st[0] - st[1]) * scale)
            hi.append((st[2] - st[0]) * scale)
        mk = MARK[m][0]
        ax.errorbar(xs, ys, yerr=[lo, hi], ls=ls, marker=mk, ms=3.5, color='black' if m == 'FRESCO' else '0.4',
                    mfc='black' if m == 'FRESCO' else 'white', mec='black', mew=0.5, elinewidth=0.5, capsize=1.2,
                    label=m)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlog:
        ax.set_xscale('log', base=2)
        ax.set_xticks(param_vals)
        ax.set_xticklabels([str(p) for p in param_vals])
    ax.grid(True, lw=0.3, color='0.88')


def fig_sensitivity():
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.6))
    _sweep_panel(axs[0, 0], 'sweep_deadline', [15.0, 20.0, 30.0, 45.0, 60.0], 'viol', 'Deadline $D$ (s)',
                 'Deadline-miss rate (%)', 100, base=30.0)
    _sweep_panel(axs[0, 1], 'sweep_load', [10.0, 20.0, 40.0], 'aoi', 'Sampling period $T_{\\mathrm{g}}$ (s)',
                 'Average age (s)', 1, base=20.0)
    _sweep_panel(axs[1, 0], 'sweep_harvest', [0.5, 1.0, 2.0], 'aoi', 'Harvest scale (relative panel area)',
                 'Average age (s)', 1, base=1.0, xlog=True)
    _sweep_panel(axs[1, 1], 'sweep_scale', [50, 100, 200, 300], 'aoi', 'Number of nodes $N$',
                 'Average age (s)', 1, base=100)
    for ax, lab in zip(axs.flat, ['(a)', '(b)', '(c)', '(d)']):
        ax.set_title(lab, loc='left', fontsize=7.5, fontweight='bold')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.0, w_pad=1.2)
    save(fig, 'fig6_sensitivity.png')


def fig_diurnal():
    tr = json.load(open(f'{OUT}/eval_trace.json'))
    r = next(x for x in tr if x['method'] == 'FRESCO' and x.get('cfg_shift_step') is None)
    rb = next(x for x in tr if x['method'] == 'FB-MPBatch' and x.get('cfg_shift_step') is None)
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.45))
    ax = axs[0]
    t = np.array(r['trace']['t']) / 3600 * 24
    ax.fill_between(t, 0, np.array(r['trace']['ghi']), color='0.88', lw=0, label='irradiance $G(t)$')
    ax.set_ylabel('Irradiance (W/m$^2$)')
    ax.set_xlabel('Time of compressed day (h)')
    ax2 = ax.twinx()
    ax2.plot(t, 100 * np.array(r['trace']['store_ch']), 'k-', lw=1.0, label='head storage, FRESCO')
    ax2.plot(np.array(rb['trace']['t']) / 3600 * 24, 100 * np.array(rb['trace']['store_ch']), 'k--', lw=0.9,
             label='head storage, FB-MPBatch')
    ax2.set_ylabel('Mean head storage (% of $C$)')
    ax2.set_ylim(0, 140)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(0, 24)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=6.0)
    ax.set_ylim(0, max(np.array(r['trace']['ghi']).max() * 1.6, 10))
    ax.set_title('(a)', loc='left', fontsize=7.5, fontweight='bold')
    ax = axs[1]
    d = np.array(r['trace']['decisions'], float)
    hrs = d[:, 0] / 3600 * 24
    bins = np.arange(0, 25, 2)
    idx = np.digitize(hrs, bins) - 1
    tau = np.array(fa.TAUS)[d[:, 1].astype(int)]
    rr = d[:, 2]
    mt = [tau[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bins) - 1)]
    mr = [rr[idx == b].mean() if (idx == b).any() else np.nan for b in range(len(bins) - 1)]
    cx = bins[:-1] + 1
    ax.plot(cx, mt, 'k-o', ms=3.5, label='mean release age $\\tau$ (s)')
    ax.set_ylabel('Release age $\\tau$ (s)')
    ax.set_xlabel('Time of compressed day (h)')
    ax.set_xlim(0, 24)
    ax.set_ylim(0, max(np.nanmax(mt) * 1.4, 5))
    ax2 = ax.twinx()
    ax2.bar(cx, mr, width=1.4, color='0.8', ec='black', lw=0.4, label='mean proposed repair level $r$', zorder=0)
    ax2.set_ylim(0, 3.2)
    ax2.set_ylabel('Proposed repair level $r$')
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=6.0)
    ax.set_title('(b)', loc='left', fontsize=7.5, fontweight='bold')
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig7_diurnal.png')


def fig_membership():
    el = fa.NeuroFuzzyElector(np.load(f'{OUT}/elector.npy'))
    prior = fa.NeuroFuzzyElector()
    names = ['storage $E_i/C$', 'harvest forecast', 'mean neighbour ETX (scaled)', 'relay load (scaled)']
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.95), sharey=True)
    x = np.linspace(0, 1, 200)
    for k, ax in enumerate(axs):
        for m, ls in zip(range(3), ['-', '--', ':']):
            ax.plot(x, np.exp(-(x - el.c[k, m]) ** 2 / (2 * el.s[k, m] ** 2)), ls=ls, color='black', lw=1.0,
                    label=['learned: low', 'learned: medium', 'learned: high'][m])
        for m, ls in zip(range(3), ['-', '--', ':']):
            ax.plot(x, np.exp(-(x - prior.c[k, m]) ** 2 / (2 * prior.s[k, m] ** 2)), ls=ls, color='0.7', lw=0.7,
                    label='prior (grey)' if m == 0 else None)
        ax.set_xlabel(names[k])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['0', '0.5', '1'])
        ax.set_ylim(0, 1.05)
        ax.grid(True, lw=0.3, color='0.9')
    axs[0].set_ylabel('Membership')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.3)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=1.2)
    save(fig, 'fig8_membership.png')


def fig_training():
    L = json.load(open(f'{OUT}/trainlog_fresco.json'))['log']
    L2 = json.load(open(f'{OUT}/trainlog_abl_nolag.json'))['log']
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.0))
    def sm(v, w=5):
        v = np.array(v, float)
        return np.convolve(v, np.ones(w) / w, mode='valid')
    ep = np.arange(len(L))
    for lg, ls, lab in [(L, '-', 'Lagrangian (FRESCO)'), (L2, '--', 'fixed penalty')]:
        e = np.arange(len(lg))
        axs[0].plot(e[4:], sm([x['aoi'] for x in lg]), ls=ls, color='black', label=lab)
        axs[1].plot(e[4:], sm([x['cost_e'] for x in lg]), ls=ls, color='black', label=lab)
    axs[1].axhline(json.load(open(f'{OUT}/selected_eps.json'))['eps_e'], color='0.6', lw=0.8, label='budget $\\epsilon_E$')
    axs[2].plot(ep, [x['lmb_e'] for x in L], 'k-', label='$\\lambda_E$')
    axs[2].plot(ep, [x['lmb_r'] for x in L], 'k:', label='$\\lambda_R$')
    axs[0].set_ylabel('Training-episode age (s)')
    axs[1].set_ylabel('Energy cost $\\hat J_E$')
    axs[2].set_ylabel('Multiplier value')
    for ax, lab in zip(axs, ['(a)', '(b)', '(c)']):
        ax.set_xlabel('Training episode')
        ax.grid(True, lw=0.3, color='0.88')
        ax.set_title(lab, loc='left', fontsize=7.5, fontweight='bold')
    axs[0].set_ylim(axs[0].get_ylim()[0], axs[0].get_ylim()[1] + 4)
    axs[0].legend(loc='upper right', fontsize=6.0)
    axs[1].set_ylim(axs[1].get_ylim()[0], 0.86)
    axs[1].legend(loc='upper right', fontsize=6.0)
    axs[2].legend(loc='center right', fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig9_training.png')


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['main', 'cd', 'cal', 'sens', 'diurnal', 'mf', 'train']
    for w, f in [('main', fig_main), ('cd', fig_cd), ('cal', fig_calibration), ('sens', fig_sensitivity),
                 ('diurnal', fig_diurnal), ('mf', fig_membership), ('train', fig_training)]:
        if w in which:
            try:
                f()
                print('ok', w)
            except Exception as e:
                import traceback
                traceback.print_exc()
