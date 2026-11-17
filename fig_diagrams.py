"""Grayscale schematic figures: system architecture (Fig. 1) and training/inference workflow (Fig. 2)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 7.5,
                     'mathtext.fontset': 'dejavuserif'})


def box(ax, x, y, w, h, text, fc='white', lw=0.9, ls='-', fs=7.2, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012,rounding_size=0.025', fc=fc, ec='black',
                       lw=lw, ls=ls, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, zorder=3,
            fontweight='bold' if bold else 'normal', linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p0, p1, ls='-', lw=0.9, rad=0.0, text=None, toff=(0, 0.02), fs=6.5, head=7):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=head, lw=lw, ls=ls, color='black',
                        connectionstyle=f'arc3,rad={rad}', zorder=4, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + toff[0], (p0[1] + p1[1]) / 2 + toff[1]
        ax.text(mx, my, text, ha='center', va='bottom', fontsize=fs, zorder=5,
                bbox=dict(fc='white', ec='none', pad=0.6))


def R(b):  # right-mid
    return (b[0] + b[2], b[1] + b[3] / 2)


def L(b):
    return (b[0], b[1] + b[3] / 2)


def T(b, f=0.5):
    return (b[0] + b[2] * f, b[1] + b[3])


def B(b, f=0.5):
    return (b[0] + b[2] * f, b[1])


# ------------------------------------------------------------------ Fig. 1
def fig_architecture(path):
    fig = plt.figure(figsize=(7.16, 3.75))
    # (a) network view
    ax = fig.add_axes([0.0, 0.0, 0.33, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 1.0)
    ax.axis('off')
    rng = np.random.default_rng(4)
    pts = rng.uniform(0.05, 0.95, size=(70, 2)) * [1, 0.82] + [0, 0.12]
    heads = np.array([[0.2, 0.78], [0.55, 0.85], [0.82, 0.62], [0.3, 0.45], [0.7, 0.35]])
    for p in pts:
        d = np.linalg.norm(heads - p, axis=1)
        h = heads[np.argmin(d)]
        ax.plot([p[0], h[0]], [p[1], h[1]], color='0.75', lw=0.4, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c='0.55', zorder=2, lw=0)
    ax.scatter(heads[:, 0], heads[:, 1], s=38, marker='s', c='black', zorder=3)
    sink = np.array([0.5, -0.04])
    ax.scatter([sink[0]], [sink[1]], s=90, marker='^', c='white', edgecolors='black', lw=1.0, zorder=4)
    ax.text(sink[0] + 0.07, sink[1] - 0.01, 'sink', fontsize=7, va='center')
    # three node-disjoint paths from head 1
    h = heads[1]
    paths = [[h, [0.48, 0.62], [0.46, 0.32], sink], [h, [0.66, 0.6], [0.62, 0.2], sink],
             [h, [0.38, 0.7], [0.18, 0.3], [0.36, 0.08], sink]]
    styles = ['-', '--', ':']
    for p, st in zip(paths, styles):
        p = np.array(p)
        ax.plot(p[:, 0], p[:, 1], ls=st, color='black', lw=1.1, zorder=3)
        ax.scatter(p[1:-1, 0], p[1:-1, 1], s=14, c='white', edgecolors='black', lw=0.7, zorder=4)
    ax.annotate('', xy=sink + [0.0, 0.035], xytext=sink + [0.0, 0.034])
    ax.text(0.55, 0.93, r'head $h$', fontsize=7, ha='left')
    # legend drawn as separate rows below the field
    ly = -0.105
    ax.scatter([0.04], [ly], s=6, c='0.55', lw=0)
    ax.text(0.07, ly, 'member', va='center', fontsize=6.5)
    ax.scatter([0.30], [ly], s=24, marker='s', c='black')
    ax.text(0.33, ly, 'head', va='center', fontsize=6.5)
    ax.scatter([0.50], [ly], s=14, c='white', edgecolors='black', lw=0.7)
    ax.text(0.53, ly, 'relay', va='center', fontsize=6.5)
    ax.text(0.72, ly, '', va='center')
    ly2 = -0.175
    for k, (st, lab) in enumerate(zip(['-', '--', ':'], [r'$\pi_1$', r'$\pi_2$', r'$\pi_3$'])):
        x0 = 0.04 + 0.25 * k
        ax.plot([x0, x0 + 0.1], [ly2, ly2], color='black', ls=st, lw=1.1)
        ax.text(x0 + 0.12, ly2, lab, va='center', fontsize=6.5)
    ax.text(0.02, 0.985, '(a)', fontsize=8, fontweight='bold', va='top')

    # (b) pipeline
    ax = fig.add_axes([0.35, 0.0, 0.65, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.0, 0.985, '(b)', fontsize=8, fontweight='bold', va='top')
    g = '0.9'
    # round level (top row)
    b1 = box(ax, 0.04, 0.82, 0.25, 0.11, 'Node state $\\mathbf{x}_i$\n(storage, harvest\nforecast, ETX, load)')
    b2 = box(ax, 0.375, 0.82, 0.25, 0.11, 'Neuro-fuzzy elector\nTSK, 81 rules\n(every round)', fc=g, bold=False)
    b3 = box(ax, 0.71, 0.82, 0.25, 0.11, 'Heads $\\mathcal{H}$, clusters $\\mathcal{M}_h$,\n$k$ disjoint min-ETX\npaths per head')
    arrow(ax, R(b1), L(b2))
    arrow(ax, R(b2), L(b3))
    ax.text(0.5, 0.955, 'Round level (every 300 s)', ha='center', fontsize=7, style='italic')
    # batch level (middle)
    b4 = box(ax, 0.04, 0.50, 0.21, 0.13, 'Head buffer\n$B$ updates,\noldest age $a$')
    b5 = box(ax, 0.30, 0.50, 0.19, 0.13, 'GAT encoder\nover head graph', fc=g)
    b6 = box(ax, 0.54, 0.50, 0.19, 0.13, 'Policy $\\pi_\\theta$\nrelease age $\\tau$\nrepair level $r$', fc=g)
    b7 = box(ax, 0.78, 0.50, 0.19, 0.13, 'Conformal shield\n$a+q_{r,t}\\,\\hat d(r)\\leq D$', fc=g)
    arrow(ax, R(b4), L(b5))
    arrow(ax, R(b5), L(b6))
    arrow(ax, R(b6), L(b7))
    arrow(ax, B(b3, 0.66), T(b7, 0.5), text=None)
    ax.plot([0.75, 0.75], [0.82, 0.70], 'k-', lw=0.9, zorder=4)
    ax.plot([0.75, 0.145], [0.70, 0.70], 'k-', lw=0.9, zorder=4)
    arrow(ax, (0.145, 0.70), T(b4, 0.5))
    ax.text(0.45, 0.715, 'Batch level (event driven)', ha='center', fontsize=7, style='italic')
    # transmission level (bottom)
    b8 = box(ax, 0.78, 0.18, 0.19, 0.15, 'Systematic RLNC\n$B$ originals + $r$\nrepairs; split $n_j$')
    b9 = box(ax, 0.48, 0.18, 0.22, 0.15, 'Duty-cycled\n802.15.4 relays\n$\\pi_1,\\ldots,\\pi_k$')
    b10 = box(ax, 0.18, 0.18, 0.22, 0.15, 'Sink decoder\ndelivery times $T_i$\nAoI $\\Delta_n(t)$')
    arrow(ax, B(b7), T(b8), text='release')
    arrow(ax, L(b8), R(b9))
    arrow(ax, L(b9), R(b10))
    # feedback: sink -> shield calibration (dashed)
    ax.plot([0.29, 0.29], [0.18, 0.07], 'k--', lw=0.9, zorder=4)
    ax.plot([0.29, 0.995], [0.07, 0.07], 'k--', lw=0.9, zorder=4)
    ax.plot([0.995, 0.995], [0.07, 0.565], 'k--', lw=0.9, zorder=4)
    arrow(ax, (0.995, 0.565), (0.97, 0.565), ls='--')
    ax.text(0.64, 0.085, 'feedback: ratio $s_i=T_i/\\hat d(r)$, ACI update of $\\alpha_{t}$', ha='center',
            fontsize=6.5, bbox=dict(fc='white', ec='none', pad=0.6))
    # AoI feedback to policy reward (dash-dot)
    ax.plot([0.24, 0.24], [0.33, 0.43], 'k-.', lw=0.8, zorder=4)
    ax.plot([0.24, 0.515], [0.43, 0.43], 'k-.', lw=0.8, zorder=4)
    arrow(ax, (0.515, 0.43), (0.635, 0.50), ls='-.', lw=0.8)
    ax.text(0.375, 0.445, 'reward $-\\Delta/D$, costs $c^E, c^R$', ha='center', fontsize=6.5,
            bbox=dict(fc='white', ec='none', pad=0.4))
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 2
def fig_training(path):
    fig, ax = plt.subplots(figsize=(7.16, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    g = '0.9'
    ax.text(0.125, 0.965, 'Stage 1: elector search (CEM)', ha='center', fontsize=7.5, fontweight='bold')
    s1 = box(ax, 0.03, 0.70, 0.19, 0.16, 'Sample $\\boldsymbol{\\phi}^{(m)}\\sim$\n$\\mathcal{N}(\\boldsymbol{\\mu},\\mathrm{diag}\\,\\boldsymbol{\\sigma}^2)$')
    s2 = box(ax, 0.03, 0.42, 0.19, 0.16, 'Episodes with fixed\nrelease rule and\nACI shield')
    s3 = box(ax, 0.03, 0.12, 0.19, 0.16, 'Elite refit of\n$(\\boldsymbol{\\mu},\\boldsymbol{\\sigma})$')
    arrow(ax, B(s1), T(s2))
    arrow(ax, B(s2), T(s3), text='fitness $F$', toff=(0.045, -0.01))
    ax.plot([0.03, 0.012, 0.012], [0.20, 0.20, 0.78], 'k-', lw=0.9, zorder=4)
    arrow(ax, (0.012, 0.78), (0.03, 0.78))
    ax.text(0.66, 0.965, 'Stage 2: Lagrangian GAT-MAPPO with the shield in the loop', ha='center', fontsize=7.5,
            fontweight='bold')
    e0 = box(ax, 0.27, 0.70, 0.14, 0.16, 'Frozen elector\n$\\boldsymbol{\\phi}^{\\star}$', fc=g)
    e7 = box(ax, 0.27, 0.42, 0.14, 0.16, 'Actor $\\pi_\\theta$\n(GAT + MLP)', fc=g)
    e1 = box(ax, 0.48, 0.70, 0.19, 0.16, 'Simulator rollout\n(1 compressed day)')
    e2 = box(ax, 0.75, 0.70, 0.22, 0.16, 'Semi-Markov buffer\n$(o,\\tau,r,R,C^E,C^R,k)$')
    e3 = box(ax, 0.75, 0.42, 0.22, 0.16, 'Critic heads\n$V_r,\\,V_E,\\,V_R$ and GAE')
    e4 = box(ax, 0.48, 0.42, 0.19, 0.16, 'Lagrangian advantage\n$\\hat A_r-\\lambda_E\\hat A_E-\\lambda_R\\hat A_R$')
    e5 = box(ax, 0.495, 0.12, 0.16, 0.16, 'Clipped PPO loss\n$\\mathcal{L}(\\theta)$', fc=g)
    e6 = box(ax, 0.7375, 0.12, 0.245, 0.16, 'Dual ascent\n$\\lambda\\leftarrow[\\lambda+\\eta_\\lambda\\,\\mathrm{clip}(\\hat J/\\epsilon-1)]_+$', fs=6.7)
    arrow(ax, R(s1), L(e0))
    ax.text(0.245, 0.80, '$\\boldsymbol{\\mu}$', ha='center', fontsize=6.5)
    arrow(ax, R(e0), L(e1))
    arrow(ax, (0.41, 0.55), (0.50, 0.70), text=None)
    ax.text(0.475, 0.605, '$\\tau, r$', fontsize=6.8, ha='center')
    arrow(ax, R(e1), L(e2))
    arrow(ax, B(e2), T(e3))
    arrow(ax, L(e3), R(e4))
    arrow(ax, B(e4), T(e5))
    arrow(ax, B(e3), T(e6), text='$\\hat J_E,\\hat J_R$', toff=(0.045, -0.015))
    arrow(ax, L(e6), R(e5))
    ax.text(0.696, 0.21, '$\\lambda_E,\\lambda_R$', ha='center', va='bottom', fontsize=6.3)
    # dashed reverse-mode gradient from the loss back to the actor parameters
    ax.plot([0.495, 0.34], [0.20, 0.20], 'k--', lw=1.0, zorder=4)
    arrow(ax, (0.34, 0.20), (0.34, 0.42), ls='--', lw=1.0)
    ax.text(0.33, 0.31, '$\\nabla_\\theta\\mathcal{L}$\n(backprop.)', ha='right', va='center', fontsize=6.5)
    ax.text(0.62, 0.035, 'Inference: deterministic arg max of both policy heads; shield and calibrator stay online',
            ha='center', fontsize=6.8, style='italic')
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    fig_architecture(os.path.join(os.environ.get('FRESCO_FIG', '.'), 'fig1_architecture.png'))
    fig_training(os.path.join(os.environ.get('FRESCO_FIG', '.'), 'fig2_training.png'))
