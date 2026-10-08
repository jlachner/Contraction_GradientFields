"""Gradient fields and their contraction on the double-pendulum torus.

    python interactiveGradientFlow_scenes.py                 interactive app
    python interactiveGradientFlow_scenes.py --present       scripted talk, clicker/arrow keys
    add --debug to print every click, recompute and redraw to the terminal
    python interactiveGradientFlow_scenes.py --snapshot DIR  PNG of every scene
    python interactiveGradientFlow_scenes.py --export-all    MP4/GIF of every scene

Requires numpy and matplotlib; ffmpeg is optional (MP4 export, else GIF).

The flow is first order, a viscous force balance

    G(q) qdot = -dV(q),        G = D_q + J_p^T D_p J_p + J_r^T D_r J_r,

so the pulled-back damping G is at the same time the metric of the gradient,
the metric of the Levi-Civita connection in Hess_G V and the metric in which
contraction is certified:

    (Df)^T G + G Df + Gdot = -2 Hess_G V,      Hess_G V = d2V - Gamma^k dV_k.

Scenes
  1 Potential    V decides where the arm goes and whether it stays there.
  2 Metric       G decides the route and which minimum a start reaches.
  3 Hessian      Hess_G V decides how perturbations around a motion respond.
  4 Contraction  curves of starts shrink at a rate certified before simulating.

Interaction: click the flat chart to move the start (scenes 1-3) or the string
(scene 4); presets jump to the characteristic cases; drag the torus to rotate;
keys 1-4 switch scenes, space pauses.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import time
import traceback

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
import matplotlib.patheffects as pe

import pendulum_gradient_model as pm

# ----------------------------------------------------------------------------
# Look
# ----------------------------------------------------------------------------

BG, PANEL, FG, MUTED, GRID = "#0d1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
ACCENT = "#ff2d55"
METRIC_NAMES = ("identity", "conformal", "damping", "inertia")
METRIC_COLORS = {"identity": "#58a6ff", "conformal": "#bc8cff",
                 "damping": "#ff2d55", "inertia": "#f2a33a"}
BASIN_COLORS = ("#2c5d8f", "#8a3a48", "#3d7046", "#6b4785")
OMEGA_COLOR = "#79c0ff"
TARGET_COLOR = "#ffd43b"
HALO = [pe.Stroke(linewidth=4.2, foreground="black", alpha=0.75), pe.Normal()]
LIGHT = np.array([-0.35, -0.55, 0.76]) / np.linalg.norm([-0.35, -0.55, 0.76])

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "legend.facecolor": PANEL, "legend.edgecolor": GRID, "legend.labelcolor": FG,
    "font.size": 10, "mathtext.fontset": "dejavusans",
})

SCENES = ("1 · Potential", "2 · Metric", "3 · Hessian", "4 · Contraction")

HEADLINES = {
    0: ("1 · The potential decides where the arm goes, and whether it stays there",
        "Minima are critical points of V, so no metric can move them. At a minimum Hess_G V = ∂²V: "
        "whether it is stable is set by V alone; the metric only scales the rate."),
    1: ("2 · The metric decides the route, and which minimum a start reaches",
        "Same V, same minima. A conformal metric c(q)·I keeps every path and only changes speed. "
        "Pulled-back damping bends the joint path until the hand moves in a straight line."),
    2: ("3 · The covariant Hessian decides how nearby motions respond",
        "d/dt |δq|²_G = −2 δqᵀ Hess_G V δq along any motion. Green: neighbours converge at "
        "rate λ_min(G⁻¹Hess_G V); red: some direction separates."),
    3: ("4 · Contraction: whole curves of starts shrink at a rate known before simulating",
        "Any curve of initial conditions obeys L(t) ≤ L(0)·exp(−∫λ_min dt). On the invariant set Ω "
        "the rate λ_Ω is an a-priori certificate. It survives a moving target, where no equilibrium exists."),
}

EQUATIONS = {
    0: r"$V = a_q V_q + a_p V_p + a_r V_r \qquad \dot q = -G(q)^{-1}\,\partial V$",
    1: r"$G = D_q + J_p^{\top} D_p J_p + J_r^{\top} D_r J_r \qquad \Gamma^k_{ij} = G^{kl}\,p_l^{\top} D_p\, p_{ij}$",
    2: r"$\mathrm{Hess}_G V = \partial^2 V - \Gamma^k\,\partial_k V \qquad \frac{d}{dt}\|\delta q\|_G^2 = -2\,\delta q^{\top}\mathrm{Hess}_G V\,\delta q$",
    3: r"$Df^{\top}G + G\,Df + \dot G = -2\,\mathrm{Hess}_G V \;\Rightarrow\; L(t) \leq L_0\, e^{-\int \lambda_{min} dt}$",
}

DEFAULT_STARTS = {0: (-1.2, -0.4), 1: (-1.2, -0.4), 2: (-1.4, -2.4)}
VIEWS = {0: (-35, -90), 1: (-35, -90), 2: (45, 125), 3: (-45, 60)}

# Presets: label -> settings applied on click
PRESETS = {
    2: {"expands, then contracts": {"start": (-1.4, -2.4), "eps": 0.10},
        "near the ridge: splits": {"start": (-2.0, -1.2), "eps": 0.06},
        "deep in the well": {"start": (1.95, -1.45), "eps": 0.25}},
    3: {"inside Ω (certified)": {"string": "omega"},
        "wide string": {"string": ((0.3, -1.0), 0.9, 31.0)},
        "across the ridge": {"string": ((0.8, 0.0), 0.5, 135.0)}},
}


# Matplotlib's default keys include q = quit, l = log scale, s = save dialog,
# p/o = pan/zoom.  In a live talk a stray key must not change or close the figure.
for _key in [k for k in plt.rcParams if k.startswith("keymap.")]:
    plt.rcParams[_key] = []
plt.rcParams["keymap.fullscreen"] = ["f", "ctrl+f"]
plt.rcParams["keymap.quit"] = ["cmd+w", "ctrl+w"]

# ----------------------------------------------------------------------------
# Presentation script: every step sets the complete state, so stepping forward
# or back always shows exactly the same picture.  Notes print to the terminal.
# ----------------------------------------------------------------------------

STEPS = [
    dict(title="Two minima, one potential", scene=0, metric="damping",
         show="flat chart: the two white stars; workspace: the two grey ghost arms",
         say=["The configuration space of the double pendulum is a torus: shoulder angle around "
              "the ring, elbow angle around the tube. The flat chart is the same torus cut open.",
              "V is a spring that pulls the hand to the yellow star. Two arm configurations reach it, "
              "elbow up and elbow down, so V has two minima.",
              "Nine starts flow along qdot = -G^-1 dV. Each one ends in one of the two minima."]),
    dict(title="The minima do not depend on the metric", scene=0, metric="identity",
         show="the stars stay put; the λ ranges in the text block change",
         say=["Same V, now with the identity metric. The paths change, the stars do not.",
              "At a critical point dV = 0, so the Christoffel term vanishes: Hess_G V = d²V. "
              "Whether a minimum is stable is decided by V alone.",
              "The metric only scales the rates: compare the λ values with the previous step."]),
    dict(title="Superposition reshapes the landscape", scene=0, metric="damping",
         sliders={"a_q": 0.5},
         show="only one star is left; every start now goes there",
         say=["Add a joint spring to the hand spring: V = a_q V_q + a_p V_p.",
              "The elbow-down minimum disappears and every start converges to one configuration.",
              "Knob 1: the potential decides where the motion goes, and whether it converges."]),
    dict(title="Same potential, four metrics", scene=1, metric="damping",
         show="workspace: the red hand path is straight; flat chart: dashed basin boundary",
         say=["Same V and the same start, integrated with four metrics at once.",
              "Red is the pulled-back damping G = D_q + J_p^T D_p J_p: the hand moves in an almost "
              "straight line (path/chord 1.01). Blue, the identity metric, swings around and ends in "
              "the other elbow solution.",
              "The dashed purple path, the conformal metric c(q)I, lies exactly on the blue one and "
              "is only slower: only the anisotropic part of G bends paths.",
              "The dashed blue line is the identity basin boundary: the metric moved the boundary, "
              "not the minima. Knob 2: the metric decides the route."]),
    dict(title="Joint damping versus task-space damping", scene=1, metric="damping",
         sliders={"d_q": 1.0},
         show="workspace: the red hand path bends away from the straight line",
         say=["Increase the joint damping D_q. The metric moves from task space towards joint space.",
              "The hand path bends again (path/chord 1.01 → 1.32) and convergence slows down "
              "(90% of the way at t = 1.4 → 3.0).",
              "The damping matrices are the design parameters of the metric."]),
    dict(title="The Hessian predicts the perturbation response", scene=2, metric="damping",
         preset="expands, then contracts",
         show="the ring on the flat chart; the band and curves in the lower-right plot",
         say=["Now a ring of neighbouring starts around one motion.",
              "Background: λ_min of G^-1 Hess_G V. Green, neighbours converge; red, some direction "
              "separates. The red band at θ2 = 0 is the stretched arm, the kinematic singularity.",
              "The ring stretches while crossing the red band, then shrinks by about five orders of "
              "magnitude in the green region. The grey band is what the Hessian predicts.",
              "Knob 3: the covariant Hessian decides how nearby motions respond."]),
    dict(title="Near the ridge, a perturbation decides the outcome", scene=2, metric="damping",
         preset="near the ridge: splits",
         show="white paths: ring members that end in the other minimum; flat curves in the plot",
         say=["Start close to the ridge between the two minima.",
              "Part of the ring ends in the other minimum. Those curves level off in the plot.",
              "This expansion is exactly what the red region of Hess_G V predicts."]),
    dict(title="Contraction, certified before simulating", scene=3, metric="damping",
         preset="inside Ω (certified)",
         show="blue region Ω; red string; three curves in the lower-right plot",
         say=["Contraction is a statement about all pairs of trajectories. So take a whole curve of "
              "starts and measure its length in G.",
              "Ω is a sublevel set of V, so the flow can never leave it. On Ω the smallest rate, "
              "λ_Ω = 0.53, is computed from V and G alone, before any simulation.",
              "The measured length (red) stays below the bound along the string (dashed) and below "
              "the a-priori certificate (dotted)."]),
    dict(title="The wrong metric gives a weaker certificate", scene=3, metric="damping",
         preset="inside Ω (certified)", euclid=True,
         show="the green region on the flat chart shrinks",
         say=["Same flow, but certified in the identity metric instead of G.",
              "The green region shrinks. The flow's own metric is the right contraction metric: "
              "Df^T G + G Df + Gdot = -2 Hess_G V holds exactly only for G."]),
    dict(title="Where contraction ends", scene=3, metric="damping", preset="across the ridge",
         show="the string stretches between both minima; its length levels off",
         say=["A string that crosses the ridge.",
              "It stretches between the two minima and its length levels off. The bound still "
              "holds: it grows, because it picks up the saddle.",
              "Contraction is a regional property, and the map shows the region."]),
    dict(title="A moving target: no equilibrium, still contraction", scene=3, metric="damping",
         preset="inside Ω (certified)", moving=True,
         show="workspace: the target circles; flat chart: the string collapses onto one orbit",
         say=["Now the target moves on a circle. There is no equilibrium any more, so no "
              "Lyapunov argument about a fixed point.",
              "Every start still collapses onto one trajectory that follows the target.",
              "This is what the exponential certificate adds in practice: bounded tracking and "
              "reproducible motion."]),
]

NEXT_KEYS = ("right", "down", "pagedown", "n", "enter")
PREV_KEYS = ("left", "up", "pageup", "p", "backspace")
HOLD_FRAMES = 60            # pause on the final frame before the animation loops (2.4 s)

# ----------------------------------------------------------------------------
# Geometry for display
# ----------------------------------------------------------------------------

def _button_reader():
    """Return a function telling whether the left mouse button is physically held.

    matplotlib's macOS backend sometimes drops the release event and then keeps
    reporting the button as held, so on macOS ask the window server directly.
    Elsewhere the state is unknown (None).
    """
    if sys.platform == "darwin":
        try:
            import ctypes
            cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
            cg.CGEventSourceButtonState.restype = ctypes.c_bool
            cg.CGEventSourceButtonState.argtypes = [ctypes.c_int32, ctypes.c_uint32]
            return lambda: bool(cg.CGEventSourceButtonState(0, 0))   # combined state, left button
        except (OSError, AttributeError):
            pass
    return lambda: None


left_button_held = _button_reader()


def torus_xyz(q, major=2.0, minor=0.75, lift=0.0):
    t1, t2 = q[..., 0], q[..., 1]
    r = minor + lift
    return np.stack(((major + r * np.cos(t2)) * np.cos(t1),
                     (major + r * np.cos(t2)) * np.sin(t1),
                     r * np.sin(t2)), axis=0)


def torus_normal(q):
    t1, t2 = q[..., 0], q[..., 1]
    return np.stack((np.cos(t2) * np.cos(t1), np.cos(t2) * np.sin(t1), np.sin(t2)), axis=-1)


def view_vector(elev, azim):
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def torus_visible_xyz(q, view, lift=0.06, back=False):
    """Embedded points on the side facing the camera (or, with back=True, the other side)."""
    q = np.asarray(q, dtype=float)
    xyz = torus_xyz(q, lift=lift)
    hidden = np.sum(torus_normal(q) * view, axis=-1) < -0.08
    blank = ~hidden if back else hidden
    if blank.any():
        xyz = xyz.copy()
        xyz[:, blank] = np.nan
    return xyz


def flat_xy(path):
    """Wrap a path into [-pi, pi)^2 and break the line where it jumps."""
    w = pm.wrap_angle(np.asarray(path, dtype=float))
    if len(w) < 2:
        return w[:, 0], w[:, 1]
    jump = np.any(np.abs(np.diff(w, axis=0)) > np.pi, axis=1)
    if jump.any():
        w = np.insert(w, np.nonzero(jump)[0] + 1, np.nan, axis=0)
    return w[:, 0], w[:, 1]


def arm_xy(q):
    elbow, tip = pm.forward_kinematics(np.asarray(q, dtype=float))
    return np.array([0.0, elbow[0], tip[0]]), np.array([0.0, elbow[1], tip[1]])


def metric_ring(center, G, radius, n=40):
    """Points at metric distance `radius` from center: {v : v^T G v = r^2}."""
    vals, vecs = np.linalg.eigh(G)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    circle = np.stack((np.cos(a), np.sin(a)), axis=-1)
    return center + radius * np.einsum("nj,ij->ni", circle, vecs / np.sqrt(vals))


def grid_index(q, n):
    return np.rint((pm.wrap_angle(q) + np.pi) / (2 * np.pi) * n).astype(int) % n


def subsample(n_total, n_frames=240):
    return np.unique(np.linspace(0, n_total - 1, min(n_total, n_frames)).astype(int))


def cumulative_rate(rate, dt):
    return np.concatenate(([0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)))


# ----------------------------------------------------------------------------
# Application
# ----------------------------------------------------------------------------

class App:
    def __init__(self, interactive=True, present=False, debug=False):
        self.interactive = interactive
        self.present = present
        self.debug = debug
        self._mouse_down = False
        self._pending_refresh = False
        self._mouse_seen = 0.0            # time of the last press
        self.step_index = 0
        self._error_reported = False
        self.scene = 3
        self.metric_name = "damping"
        self.euclid = False
        self.moving = False
        self.starts = {k: np.array(v, dtype=float) for k, v in DEFAULT_STARTS.items()}
        self.string = None              # (centre, half, angle_deg); None = inside Omega
        self.frame = 0
        self.playing = True
        self._cache = {}
        self._view = VIEWS[self.scene]
        self._view_used = None
        self._last_n = 0
        self._timer = None
        self._busy = True                 # no animation frames until a scene is built

        self.potential = pm.Potential(a_q=0.0, a_p=1.0, a_r=0.0, p0=(0.9, -0.9), k_p=2.0)
        self.metrics = {"identity": pm.Identity(), "conformal": pm.Conformal(),
                        "damping": pm.Damping(), "inertia": pm.Inertia()}

        self.n_grid = 121
        self.grid = np.linspace(-np.pi, np.pi, self.n_grid, endpoint=False)
        self.Q = np.stack(np.meshgrid(self.grid, self.grid, indexing="ij"), axis=-1)

        if present:
            plt.rcParams["toolbar"] = "None"      # no pan/zoom buttons to hit by accident
        self._build_figure()
        if present:
            self.warm_up()
            self.apply_step(0)
        else:
            self.refresh()

    # ------------------------------------------------------------- presenting
    def _quiet(self, widget, action):
        """Change a widget's state without firing callbacks or redraws."""
        widget.eventson, widget.drawon = False, False
        try:
            action()
        finally:
            widget.eventson, widget.drawon = True, True

    def _configure(self, step):
        """Set the complete state of one presentation step (no drawing)."""
        for name, s in self.sliders.items():
            self._set_quietly(name, s.valinit)
        for name, value in step.get("sliders", {}).items():
            self._set_quietly(name, value)
        self.starts = {k: np.array(v, dtype=float) for k, v in DEFAULT_STARTS.items()}
        self.string = None
        self.scene = step["scene"]
        self.metric_name = step["metric"]
        self.euclid = step.get("euclid", False)
        self.moving = step.get("moving", False)
        self._view = VIEWS[self.scene]
        self._quiet(self.r_scene, lambda: self.r_scene.set_active(self.scene))
        self._quiet(self.r_metric, lambda: self.r_metric.set_active(METRIC_NAMES.index(self.metric_name)))
        status = self.c_opts.get_status()
        for i, want in enumerate((self.euclid, self.moving)):
            if status[i] != want:
                self._quiet(self.c_opts, lambda i=i: self.c_opts.set_active(i))
        label = step.get("preset")
        if label is not None:
            cfg = PRESETS[self.scene][label]
            if "start" in cfg:
                self.starts[self.scene] = np.array(cfg["start"], dtype=float)
            if "eps" in cfg:
                self._set_quietly("eps", cfg["eps"])
            if "string" in cfg and cfg["string"] != "omega":
                centre, half, angle = cfg["string"]
                self.string = (np.array(centre, dtype=float), half, angle)
                self._set_quietly("half", half)
                self._set_quietly("angle", angle)
            r = self.r_presets[self.scene]
            self._quiet(r, lambda: r.set_active(list(PRESETS[self.scene]).index(label)))

    def warm_up(self):
        """Compute every step once, so that stepping during the talk never waits."""
        print(f"Preparing {len(STEPS)} presentation steps ...", flush=True)
        for i, step in enumerate(STEPS):
            self._configure(step)
            self.refresh()
            self._render_frame(self.n_frames // 2)
            print(f"  ready {i + 1:2d}/{len(STEPS)}  {step['title']}", flush=True)
        print("Ready. Next: → ↓ PageDown n Enter   Back: ← ↑ PageUp p   "
              "Pause: space   Restart: r   Full screen: f\n", flush=True)

    def apply_step(self, index):
        index = int(np.clip(index, 0, len(STEPS) - 1))
        step = STEPS[index]
        try:
            self._configure(step)
            self.refresh()
        except Exception:
            traceback.print_exc()
            print("Step could not be shown; staying on the previous one.", flush=True)
            return
        self.step_index = index
        self.frame = 0
        self.playing = True
        self.b_play.label.set_text("Pause")
        self.step_text.set_text(f"{index + 1} / {len(STEPS)}  ·  {step['title']}")
        self._print_notes(index)

    @staticmethod
    def _print_notes(index):
        step = STEPS[index]
        bar = "─" * 78
        print(bar)
        print(f"STEP {index + 1}/{len(STEPS)}  ·  {SCENES[step['scene']]}  ·  {step['title']}")
        print(f"LOOK AT: {step['show']}")
        for line in step["say"]:
            print(f"  • {line}")
        if index + 1 < len(STEPS):
            print(f"NEXT:    {STEPS[index + 1]['title']}")
        print(bar, flush=True)

    # ------------------------------------------------------------------ layout
    def _build_figure(self):
        fig = self.fig = plt.figure(figsize=(16, 9.4))
        self.title = fig.text(0.015, 0.962, "", fontsize=15.5, fontweight="bold")
        self.subtitle = fig.text(0.015, 0.928, "", fontsize=10.2, color=MUTED)
        self.ax3 = fig.add_axes([0.0, 0.25, 0.42, 0.66], projection="3d")
        self.axflat = fig.add_axes([0.445, 0.36, 0.25, 0.52])
        self.axws = fig.add_axes([0.73, 0.50, 0.25, 0.39])
        self.axplot = fig.add_axes([0.765, 0.075, 0.215, 0.33])
        self.status = fig.text(0.445, 0.232, "", fontsize=8.4, va="bottom", family="monospace",
                               color=FG)
        self.step_text = fig.text(0.985, 0.985, "", fontsize=9.5, ha="right", va="top", color=MUTED)
        self.equation = fig.text(0.21, 0.262, "", fontsize=12.5, ha="center", va="center",
                                 bbox=dict(boxstyle="round,pad=0.45", fc=PANEL, ec=GRID))

        def panel(rect, title):
            ax = fig.add_axes(rect, facecolor=PANEL)
            ax.set_title(title, fontsize=9, color=MUTED)
            for sp in ax.spines.values():
                sp.set_color(GRID)
            return ax

        def style_radio(r):
            for lbl in r.labels:
                lbl.set_color(FG)
                lbl.set_fontsize(9.5)

        # useblit=False: the widgets must not blit on their own on top of the scene blitting
        self.r_scene = RadioButtons(panel([0.012, 0.02, 0.115, 0.19], "Scene"), SCENES,
                                    active=self.scene, activecolor=ACCENT, useblit=False)
        style_radio(self.r_scene)
        self.r_scene.on_clicked(self._on_scene)

        self.r_metric = RadioButtons(panel([0.135, 0.02, 0.1, 0.19], "Metric G"), METRIC_NAMES,
                                     active=METRIC_NAMES.index(self.metric_name), activecolor=ACCENT,
                                     useblit=False)
        style_radio(self.r_metric)
        for lbl, name in zip(self.r_metric.labels, METRIC_NAMES):
            lbl.set_color(METRIC_COLORS[name])
        self.r_metric.on_clicked(self._on_metric)

        self.c_opts = CheckButtons(panel([0.245, 0.02, 0.165, 0.08], ""),
                                   ("certify in identity metric", "moving target"),
                                   (self.euclid, self.moving), useblit=False)
        for lbl in self.c_opts.labels:
            lbl.set_color(FG)
            lbl.set_fontsize(9)
        self.c_opts.on_clicked(self._on_option)

        self.b_play = Button(fig.add_axes([0.245, 0.11, 0.08, 0.035]), "Pause",
                             color=PANEL, hovercolor=GRID)
        self.b_play.label.set_color(FG)
        self.b_play.on_clicked(self._on_play)
        self.b_export = Button(fig.add_axes([0.33, 0.11, 0.08, 0.035]), "Export video",
                               color=PANEL, hovercolor=GRID)
        self.b_export.label.set_color(FG)
        self.b_export.on_clicked(lambda _: self.export())

        self.r_presets = {}
        for scene, presets in PRESETS.items():
            r = RadioButtons(panel([0.245, 0.155, 0.165, 0.085], "Presets"), tuple(presets),
                             active=0, activecolor=ACCENT, useblit=False)
            style_radio(r)
            r.on_clicked(lambda label, sc=scene: self._on_preset(sc, label))
            self.r_presets[scene] = r

        slots = [0.185, 0.135, 0.085, 0.035]

        def slider(slot, label, lo, hi, val, scenes, fmt="%.2f"):
            a = fig.add_axes([0.505, slots[slot], 0.16, 0.022], facecolor=PANEL)
            s = Slider(a, label, lo, hi, valinit=val, valfmt=fmt, color=ACCENT, track_color=GRID)
            s.label.set_fontsize(8.5)
            s.label.set_color(FG)
            s.valtext.set_color(FG)
            s.on_changed(lambda _: self._schedule_refresh())
            s.scenes = scenes
            return s

        P = self.potential
        self.sliders = {
            "a_q": slider(0, "joint a_q", 0.0, 1.5, P.a_q, {0}),
            "a_r": slider(1, "orient. a_r", 0.0, 1.5, P.a_r, {0}),
            "px": slider(2, "target x", -1.8, 1.8, P.p0[0], {0}),
            "py": slider(3, "target y", -1.8, 1.8, P.p0[1], {0}),
            "d_q": slider(0, "D_q", 0.02, 1.0, 0.15, {1, 2, 3}),
            "aniso": slider(1, "D_p x:y (log₂)", -2.0, 2.0, 0.0, {1}),
            "d_r": slider(2, "D_r", 0.0, 2.0, 0.0, {1}),
            "eps": slider(1, "perturbation ε", 0.01, 0.6, 0.10, {2}),
            "half": slider(1, "string ½-length", 0.05, 1.2, 0.3, {3}),
            "angle": slider(2, "string angle °", 0.0, 180.0, 135.0, {3}, fmt="%.0f"),
            "radius": slider(3, "target circle r", 0.0, 0.6, 0.35, {3}),
        }
        fig.canvas.mpl_connect("button_press_event", self._on_press)
        fig.canvas.mpl_connect("button_release_event", self._on_release)
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.dynamic = []
        self._bg = None
        if self.interactive:
            # Blitting: the static scene is drawn once, each frame only redraws moving artists.
            fig.canvas.mpl_connect("draw_event", self._on_draw)
            self._anim_timer = fig.canvas.new_timer(interval=40)
            self._anim_timer.add_callback(self._animate, None)
            self._anim_timer.start()
            fig.canvas.mpl_connect("close_event", self._on_close)
            self._last_tick = None             # set by the first animation frame
            # Always on: with the macOS backend the window hung after a click whenever this
            # thread was absent (--present froze, --present --debug never did).
            threading.Thread(target=self._watchdog, daemon=True).start()

    # ---------------------------------------------------------------- blitting
    def _dyn(self, artist):
        """Register an artist that changes every frame (interactive mode only)."""
        if self.interactive:
            artist.set_animated(True)
            self.dynamic.append(artist)
        return artist

    def _watchdog(self):
        """If the animation timer stops ticking, print where the program is stuck."""
        reported = False
        while True:
            time.sleep(0.5)
            if self._last_tick is None:        # still preparing at startup
                continue
            silent = time.time() - self._last_tick
            if silent > 2.0 and not reported:
                print(f"\n!! FROZEN: animation timer silent for {silent:.1f} s "
                      f"(mouse_down={self._mouse_down}, busy={self._busy}). Stacks:", flush=True)
                faulthandler.dump_traceback(all_threads=True)
                reported = True
            elif silent < 1.0 and reported:
                print("!! unfrozen", flush=True)
                reported = False

    def _log(self, msg):
        if self.debug:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def _on_draw(self, _event):
        """After any full redraw: keep the static background, overlay the moving artists."""
        self._log("full redraw")
        if self._busy:
            return
        canvas = self.fig.canvas
        self._bg = canvas.copy_from_bbox(self.fig.bbox)
        self._draw_dynamic()

    def _draw_dynamic(self):
        for a in self.dynamic:
            a.axes.draw_artist(a)
        self.fig.canvas.blit(self.fig.bbox)

    # --------------------------------------------------------------- callbacks
    def _on_close(self, _event):
        self._anim_timer.stop()
        if self._timer is not None:
            self._timer.stop()

    def _on_scene(self, label):
        self.scene = SCENES.index(label)
        self._view = VIEWS[self.scene]
        self._request_refresh()

    def _on_metric(self, label):
        self.metric_name = label
        self._request_refresh()

    def _on_option(self, label):
        if label.startswith("certify"):
            self.euclid = not self.euclid
        else:
            self.moving = not self.moving
        self._request_refresh()

    def _on_play(self, _=None):
        self.playing = not self.playing
        self.b_play.label.set_text("Pause" if self.playing else "Play")

    def _on_key(self, event):
        key = event.key
        if key == " ":
            self._on_play()
        elif key == "r":
            self.frame = 0
            self.playing = True
            self.b_play.label.set_text("Pause")
        elif self.present and key in NEXT_KEYS:
            self.apply_step(self.step_index + 1)
        elif self.present and key in PREV_KEYS:
            self.apply_step(self.step_index - 1)
        elif key in ("1", "2", "3", "4"):
            if self.present:      # jump to the first step of that scene
                self.apply_step(next(i for i, s in enumerate(STEPS) if s["scene"] == int(key) - 1))
            else:
                self.r_scene.set_active(int(key) - 1)

    def _on_preset(self, scene, label):
        cfg = PRESETS[scene][label]
        if "start" in cfg:
            self.starts[scene] = np.array(cfg["start"], dtype=float)
        if "eps" in cfg:
            self._set_quietly("eps", cfg["eps"])
        if "string" in cfg:
            if cfg["string"] == "omega":
                self.string = None
            else:
                centre, half, angle = cfg["string"]
                self.string = (np.array(centre, dtype=float), half, angle)
                self._set_quietly("half", half)
                self._set_quietly("angle", angle)
        self._request_refresh()

    def _axes_name(self, ax):
        for name, a in (("torus", self.ax3), ("flat chart", self.axflat), ("workspace", self.axws),
                        ("plot", self.axplot), ("scene buttons", self.r_scene.ax),
                        ("metric buttons", self.r_metric.ax), ("options", self.c_opts.ax)):
            if ax is a:
                return name
        for name, sl in self.sliders.items():
            if ax is sl.ax:
                return f"slider {name}"
        return "presets" if any(ax is r.ax for r in self.r_presets.values()) else str(ax)

    def _on_press(self, event):
        # While a mouse button is held the animation stops repainting, so a
        # dragged slider or a rotated torus is never painted over.
        self._mouse_down = True
        self._mouse_seen = time.time()
        self._log(f"press   on {self._axes_name(event.inaxes)}")
        self._on_click(event)

    def _on_release(self, event):
        self._log(f"release on {self._axes_name(event.inaxes)}")
        self._release()

    def _release(self, missed=None):
        """End a press. The macOS backend sometimes drops the release event; without
        this the animation stayed paused and a slider kept grabbing every later click."""
        if missed:
            self._log(f"release missed ({missed}): recovering")
        self._mouse_down = False
        canvas = self.fig.canvas
        if missed:
            canvas._button = None            # matplotlib would still report the button as held
        for s in self.sliders.values():
            if s.drag_active:
                s.drag_active = False
                canvas.release_mouse(s.ax)
        if self._pending_refresh:            # a slider was moved: recompute once, now
            self._pending_refresh = False
            self._request_refresh()

    def _on_click(self, event):
        if event.inaxes is not self.axflat or event.xdata is None or event.button != 1:
            return
        q = np.array([event.xdata, event.ydata])
        if self.scene == 3:
            self.string = (q, self.sliders["half"].val, self.sliders["angle"].val)
        else:
            self.starts[self.scene] = q
        self._request_refresh()

    def _set_quietly(self, name, value):
        """Move a slider without triggering a refresh or a redraw."""
        s = self.sliders[name]
        s.eventson, s.drawon = False, False
        s.set_val(value)
        s.eventson, s.drawon = True, True

    def _schedule_refresh(self):
        """A slider moved: recompute when the mouse is released (immediately otherwise)."""
        if self.scene == 3 and getattr(self, "_last_string", None) is not None:
            centre, half, angle = self._last_string
            new = (self.sliders["half"].val, self.sliders["angle"].val)
            if abs(new[0] - half) > 1e-9 or abs(new[1] - angle) > 1e-9:
                self.string = (centre, *new)     # the user reshaped the string: keep its centre
        if self.interactive and self._mouse_down:
            self._pending_refresh = True
        else:
            self._request_refresh()

    # ------------------------------------------------------------------- state
    def _sync_parameters(self):
        s = {k: v.val for k, v in self.sliders.items()}
        P = self.potential
        P.a_q, P.a_r = s["a_q"], s["a_r"]
        P.p0 = np.array([s["px"], s["py"]])
        D = self.metrics["damping"]
        D.d_q, D.d_r = s["d_q"], s["d_r"]
        r = 2.0 ** s["aniso"]
        D.D_p = np.diag([np.sqrt(r), 1.0 / np.sqrt(r)])

    @property
    def q_start(self):
        return self.starts.get(self.scene, self.starts[0])

    @property
    def metric(self):
        return self.metrics[self.metric_name]

    def _pkey(self):
        P = self.potential
        return (round(P.a_q, 4), round(P.a_r, 4), round(P.p0[0], 4), round(P.p0[1], 4))

    def _mkey(self, name=None):
        name = name or self.metric_name
        if name != "damping":
            return name
        D = self.metrics["damping"]
        return (name, round(D.d_q, 4), round(D.d_r, 4), round(D.D_p[0, 0], 4))

    def _cached(self, key, fn):
        if key not in self._cache:
            if len(self._cache) > 400:
                self._cache.clear()
            self._cache[key] = fn()
        return self._cache[key]

    def _memo(self, fn, *args, **kw):
        """Cache trajectories, so revisiting a presentation step is instant."""
        def part(a):
            if isinstance(a, np.ndarray):
                return (a.shape, a.tobytes())
            if isinstance(a, pm.Potential):
                extra = (a.radius, a.omega) if isinstance(a, pm.MovingTarget) else None
                return ("P", self._pkey(), extra)
            if isinstance(a, pm.Metric):
                return ("M", self._mkey(next(n for n, m in self.metrics.items() if m is a)))
            return a
        key = (fn.__name__,) + tuple(part(a) for a in args) + tuple(sorted(kw.items()))
        return self._cached(key, lambda: fn(*args, **kw))

    def dt(self, name=None):
        """RK4 step from the stiffest generalised eigenvalue of the flow on the torus."""
        name = name or self.metric_name

        def compute():
            H, G = pm.covariant_hessian(self.Q, self.potential, self.metrics[name])
            lo, hi = pm.generalized_eigs(H, G)
            stiff = max(np.abs(lo).max(), np.abs(hi).max())
            return float(np.clip(1.6 / stiff, 0.002, 0.02))
        return self._cached(("dt", self._pkey(), self._mkey(name)), compute)

    def minima(self):
        return self._cached(("minima", self._pkey()),
                            lambda: pm.find_minima(self.potential, self.metrics["identity"],
                                                   n=16, duration=40.0, dt=0.03))

    def rate_grid(self, euclid=None):
        euclid = self.euclid if euclid is None else euclid
        fn = pm.euclidean_rate if euclid else pm.contraction_rate
        return self._cached(("rate", self._pkey(), self._mkey(), euclid),
                            lambda: fn(self.Q, self.potential, self.metric))

    def basins(self, name):
        mins = self.minima()
        slow = name == "conformal"           # c(q) >= 1 slows the flow down
        return self._cached(("basin", self._pkey(), self._mkey(name)),
                            lambda: pm.basin_labels(self.potential, self.metrics[name], mins, n=72,
                                                    duration=60.0 if slow else 25.0,
                                                    dt=0.05 if slow else self.dt(name))[1])

    # ----------------------------------------------------------------- drawing
    def _request_refresh(self):
        """Recompute shortly after the last of a burst of clicks, not once per click.

        A recompute takes up to a second; clicks made meanwhile queue up, and running
        one recompute per click froze the window for many seconds.
        """
        if not self.interactive:
            self.refresh()
            return
        if self._timer is None:
            self._timer = self.fig.canvas.new_timer(interval=120)
            self._timer.single_shot = True
            self._timer.add_callback(self.refresh)
        self._timer.stop()                # restart: only the last request survives
        self._timer.start()

    def refresh(self):
        t0 = time.time()
        self._busy = True
        try:
            self._refresh()
        finally:
            self._busy = False
        self._log(f"recomputed scene {self.scene + 1} ({self.metric_name}) in {time.time() - t0:.2f} s")
        if self.interactive:
            self.fig.canvas.draw()        # draw now; do not rely on an idle callback
        else:
            self.fig.canvas.draw_idle()

    def _refresh(self):
        self._sync_parameters()
        # Widgets of different scenes share screen positions: hidden ones must also be
        # inactive, or they still react to clicks and steal the mouse from the visible one.
        for s in self.sliders.values():
            on = self.scene in s.scenes
            s.ax.set_visible(on)
            s.active = on
        for scene, r in self.r_presets.items():
            r.ax.set_visible(scene == self.scene)
            r.active = scene == self.scene   # (RadioButtons.set_active selects a button)
        for ax in (self.ax3, self.axflat, self.axws, self.axplot):
            ax.cla()
        self.torus_lines = []        # (artist, back artist, path, lift, reveal)
        self.dynamic = []
        self._bg = None
        self.frame = 0
        head, sub = HEADLINES[self.scene]
        self.title.set_text(head)
        self.subtitle.set_text(sub)
        self.equation.set_text(EQUATIONS[self.scene])
        (self._scene_potential, self._scene_metric,
         self._scene_hessian, self._scene_contraction)[self.scene]()
        self._style_axes()
        self._render_frame(0)

    def _style_axes(self):
        ax = self.ax3
        ax.set_facecolor(BG)
        ax.set_box_aspect((1, 1, 0.42))
        ax.set_axis_off()
        ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6); ax.set_zlim(-1.1, 1.1)
        ax.view_init(*self._view)
        a = self.axflat
        a.set_xlim(-np.pi, np.pi); a.set_ylim(-np.pi, np.pi)
        a.set_aspect("equal")
        a.set_xlabel(r"$\theta_1$  shoulder"); a.set_ylabel(r"$\theta_2$  elbow")
        ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
        labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
        a.set_xticks(ticks); a.set_xticklabels(labels)
        a.set_yticks(ticks); a.set_yticklabels(labels)
        w = self.axws
        w.set_xlim(-2.2, 2.2); w.set_ylim(-2.2, 2.2); w.set_aspect("equal")
        w.grid(alpha=0.5, lw=0.5)
        w.set_title("Workspace", fontsize=10, color=MUTED)
        w.plot([-0.12, 0.12], [0.0, 0.0], color=MUTED, lw=3, zorder=3)
        w.plot([0], [0], "o", color=FG, ms=5, zorder=4)
        self.axplot.grid(alpha=0.6, lw=0.5)
        for ax in (self.axflat, self.axws, self.axplot):
            for sp in ax.spines.values():
                sp.set_color(GRID)

    # ------------------------------------------------------ drawing primitives
    def _surface(self, values, cmap, norm):
        n = 60
        u = np.linspace(-np.pi, np.pi, n)
        U, V = np.meshgrid(u, u, indexing="ij")
        q = np.stack((U, V), axis=-1)
        X, Y, Z = torus_xyz(q)
        idx = grid_index(q, self.n_grid)
        rgba = np.asarray(cmap(norm(values[idx[..., 0], idx[..., 1]])))
        shade = 0.62 + 0.38 * np.clip(np.sum(torus_normal(q) * LIGHT, axis=-1), 0, 1)
        rgba[..., :3] *= shade[..., None]
        self.ax3.computed_zorder = False
        self.ax3.plot_surface(X, Y, Z, facecolors=rgba, linewidth=0, antialiased=False,
                              shade=False, rcount=n, ccount=n, zorder=1)

    def _image(self, values, cmap, norm):
        self.axflat.imshow(values.T, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi),
                           cmap=cmap, norm=norm, interpolation="bilinear", aspect="auto", zorder=0)

    def _torus_path(self, path, color, lw=2.5, lift=0.06, ls="-", z=10, halo=True, alpha=1.0,
                    reveal=False):
        """A path on the torus; `reveal` draws it progressively with the animation.

        The part on the far side of the tube is drawn faint, as if seen through it.
        """
        line, = self.ax3.plot([], [], [], color=color, lw=lw, ls=ls, zorder=z, alpha=alpha,
                              path_effects=HALO if halo else None)
        back, = self.ax3.plot([], [], [], color=color, lw=max(lw * 0.5, 0.6), ls=ls, zorder=2,
                              alpha=0.22 * alpha)
        self._dyn(back)
        self._dyn(line)
        self.torus_lines.append((line, back, np.asarray(path), lift, reveal))
        return line

    def _update_torus_lines(self, n):
        view = view_vector(*self._view)
        for line, back, path, lift, reveal in self.torus_lines:
            p = path[: max(n + 1, 2)] if reveal else path
            line.set_data_3d(*torus_visible_xyz(p, view, lift))
            if back is not None:
                back.set_data_3d(*torus_visible_xyz(p, view, lift, back=True))

    def _torus_marker(self, color, ms=8, z=14):
        m, = self.ax3.plot([], [], [], "o", color=color, ms=ms, mec="black", zorder=z)
        return self._dyn(m)

    def _place(self, marker, q, lift=0.09):
        marker.set_data_3d(*torus_visible_xyz(np.asarray(q)[None], view_vector(*self._view), lift))

    def _flat_path(self, path, color, lw=2.4, ls="-", z=7, halo=False, alpha=1.0):
        line, = self.axflat.plot(*flat_xy(path), color=color, lw=lw, ls=ls, zorder=z, alpha=alpha,
                                 path_effects=HALO if halo else None)
        return line

    def _minimum_markers(self):
        mins = self.minima()
        for q in mins:
            self.axflat.plot(*q, marker="*", ms=18, color="white", mec="black", zorder=9)
            star, = self.ax3.plot([], [], [], marker="*", ms=16, color="white", mec="black",
                                  zorder=15, ls="")
            self._dyn(star)
            self.torus_lines.append((star, None, q[None], 0.11, False))
        return mins

    def _ghost_arms(self, qs, alpha=0.22):
        for q in qs:
            x, y = arm_xy(q)
            self.axws.plot(x, y, "-", color=MUTED, lw=7, alpha=alpha, solid_capstyle="round", zorder=2)

    def _arm(self, color, lw=3.2, alpha=1.0, z=8):
        line, = self.axws.plot([], [], "-o", color=color, lw=lw, ms=5.5, alpha=alpha,
                               mec="black", zorder=z, solid_capstyle="round")
        return self._dyn(line)

    def _target(self, p=None):
        p = self.potential.p0 if p is None else p
        mk, = self.axws.plot([p[0]], [p[1]], marker="*", ms=17, color=TARGET_COLOR,
                             mec="black", zorder=7, ls="")
        return mk

    def _cursor(self):
        return self._dyn(self.axplot.axvline(0, color=MUTED, lw=1, ls=":"))

    def _plot_labels(self, title, ylabel):
        self.axplot.set_title(title, fontsize=10, color=MUTED)
        self.axplot.set_xlabel("time")
        self.axplot.set_ylabel(ylabel)

    # ------------------------------------------------------------- scene 1
    def _scene_potential(self):
        P, M = self.potential, self.metric
        V = P.terms(self.Q)[0]
        norm = Normalize(V.min(), np.percentile(V, 97))
        self._surface(V, plt.cm.viridis, norm)
        self._image(V, "viridis", norm)
        self.axflat.contour(self.grid, self.grid, V.T, levels=14, colors="white",
                            linewidths=0.5, alpha=0.4)
        mins = self._minimum_markers()
        hero_c = METRIC_COLORS[self.metric_name]
        self.axflat.set_title(f"V(q), its minima, and paths under G = {self.metric_name}",
                              fontsize=10, color=MUTED)

        starts = np.array([self.q_start, [-2.6, -2.4], [-2.4, 0.6], [-0.9, 2.7], [0.6, -2.8],
                           [2.6, 1.0], [2.8, -1.4], [1.0, 1.2], [-1.6, 1.8]])
        t, paths = self._memo(pm.integrate, starts, P, M, duration=12.0, dt=self.dt())
        flat_lines = []
        for j in range(len(starts)):
            hero = j == 0
            c = hero_c if hero else "white"
            self._flat_path(paths[:, j], c, lw=1.0, alpha=0.25, z=5)
            flat_lines.append(self._dyn(self._flat_path(paths[:1, j], c, lw=2.8 if hero else 1.4,
                                                        z=7 if hero else 6, halo=hero)))
            self._torus_path(paths[:, j], c, lw=0.8, alpha=0.3, halo=False, z=8)
            self._torus_path(paths[:, j], c, lw=2.8 if hero else 1.3, z=11 if hero else 9,
                             halo=hero, reveal=True)
        self.axflat.plot(*starts.T, "o", color="white", mec="black", ms=4.5, zorder=8, ls="")

        self._ghost_arms(mins)
        if P.a_r > 0:
            p0 = P.p0
            self.axws.plot([p0[0], p0[0] + 0.4 * np.sin(P.phi0)], [p0[1], p0[1] - 0.4 * np.cos(P.phi0)],
                           color=TARGET_COLOR, lw=3, zorder=6)
        self._target()
        _, tips = pm.forward_kinematics(paths)
        hand = self._dyn(self.axws.plot([], [], color=hero_c, lw=1.8, zorder=5)[0])
        arms = [self._arm(hero_c if j == 0 else "white", lw=3.2 if j == 0 else 1.6,
                          alpha=1.0 if j == 0 else 0.35, z=8 if j == 0 else 6)
                for j in range(len(starts))]

        Vt = P.terms(paths)[0] - V.min()
        for j in range(len(starts)):
            self.axplot.semilogy(t, np.maximum(Vt[:, j], 1e-9), color=hero_c if j == 0 else MUTED,
                                 lw=2.4 if j == 0 else 0.9, alpha=1 if j == 0 else 0.7)
        self.axplot.set_ylim(1e-6, Vt.max() * 3)
        self._plot_labels("Every start descends V", r"$V(q(t)) - V_{min}$")

        lines = []
        for q in mins:
            H, G = pm.covariant_hessian(q, P, M)
            lo, hi = pm.generalized_eigs(H, G)
            stable = np.all(np.linalg.eigvalsh(P.terms(q)[2]) > 0)
            lines.append(f"  ({q[0]:+.2f},{q[1]:+.2f})  ∂²V {'≻ 0' if stable else 'indef.'}  "
                         f"λ ∈ [{float(lo):.2f}, {float(hi):.2f}]")
        self.status.set_text(f"{len(mins)} {'minimum' if len(mins) == 1 else 'minima'} of V — "
                             "the same under every metric.\n"
                             f"Rates of G⁻¹Hess_G V at the minima, G = {self.metric_name}:\n"
                             + "\n".join(lines)
                             + "\nSwitch G: minima and their stability stay, rates change.")

        frames = subsample(len(t))
        markers = [self._torus_marker(hero_c if j == 0 else "white", ms=8 if j == 0 else 5)
                   for j in range(len(starts))]
        heads = self._dyn(self.axflat.plot([], [], "o", color="white", mec="black", ms=5, zorder=10,
                                           ls="")[0])
        cur = self._cursor()

        def step(k):
            n = frames[k]
            for j in range(len(starts)):
                flat_lines[j].set_data(*flat_xy(paths[: n + 1, j]))
                arms[j].set_data(*arm_xy(paths[n, j]))
                self._place(markers[j], paths[n, j])
            w = pm.wrap_angle(paths[n])
            heads.set_data(w[:, 0], w[:, 1])
            hand.set_data(tips[: n + 1, 0, 0], tips[: n + 1, 0, 1])
            cur.set_xdata([t[n], t[n]])
            return n
        self.n_frames, self.step = len(frames), step

    # ------------------------------------------------------------- scene 2
    def _scene_metric(self):
        P = self.potential
        mins = self.minima()
        lab = self.basins(self.metric_name)
        n_lab = lab.shape[0]
        cmap = ListedColormap([GRID] + list(BASIN_COLORS[:max(len(mins), 1)]))
        norm = Normalize(-0.5, len(mins) + 0.5)
        self.axflat.imshow(lab.T + 1, origin="lower", extent=(-np.pi, np.pi, -np.pi, np.pi),
                           cmap=cmap, norm=norm, interpolation="nearest", aspect="auto", zorder=0)
        idx = np.rint((self.grid + np.pi) / (2 * np.pi) * (n_lab - 1)).astype(int)
        self._surface(lab[np.ix_(idx, idx)] + 1.0, cmap, norm)
        self._minimum_markers()
        g_lab = np.linspace(-np.pi, np.pi, n_lab)
        if self.metric_name != "identity":
            ref = self.basins("identity")
            self.axflat.contour(g_lab, g_lab, ref.T.astype(float), levels=np.arange(len(mins)) + 0.5,
                                colors=METRIC_COLORS["identity"], linewidths=1.6, linestyles="--")

        Gfun = self.metric.G
        cg = np.linspace(-np.pi, np.pi, 6, endpoint=False) + np.pi / 6
        for a in cg:
            for b in cg:
                q = np.array([a, b])
                Gq = Gfun(q)
                ring = metric_ring(q, Gq / np.max(np.linalg.eigvalsh(Gq)), 0.24)
                self.axflat.fill(ring[:, 0], ring[:, 1], facecolor="none",
                                 edgecolor=METRIC_COLORS[self.metric_name], lw=0.9, alpha=0.75)
        self.axflat.set_title(f"Basins under G = {self.metric_name} (dashed: identity)\n"
                              "ellipses: shape of the unit ball of G", fontsize=9.5, color=MUTED)

        # integrate all four metrics on one common clock
        dt = min(self.dt(n) for n in METRIC_NAMES)
        paths = {}
        for name in METRIC_NAMES:
            ts, paths[name] = self._memo(pm.integrate, self.q_start, P, self.metrics[name],
                                         duration=12.0, dt=dt)
        self._ghost_arms(mins, alpha=0.15)
        self._target()
        stats, flat_lines, hands, arms, markers = [], {}, {}, {}, {}
        for name in METRIC_NAMES:
            hero = name == self.metric_name
            c = METRIC_COLORS[name]
            ls = (0, (4, 2)) if name == "conformal" else "-"
            p = paths[name]
            self._flat_path(p, c, lw=1.0, alpha=0.3, ls=ls, z=5)
            flat_lines[name] = self._dyn(self._flat_path(p[:1], c, lw=3.0 if hero else 1.8, ls=ls,
                                                         z=8 if hero else 7, halo=hero))
            self._torus_path(p, c, lw=0.9, alpha=0.3, ls=ls, halo=False, z=8)
            self._torus_path(p, c, lw=3.0 if hero else 1.8, ls=ls, z=12 if hero else 10,
                             halo=hero, reveal=True)
            _, tip = pm.forward_kinematics(p)
            self.axws.plot(tip[:, 0], tip[:, 1], color=c, lw=0.9, alpha=0.35, ls=ls)
            hands[name] = (self._dyn(self.axws.plot([], [], color=c, lw=2.8 if hero else 1.6, ls=ls,
                                                    label=name, zorder=6)[0]), tip)
            arms[name] = self._arm(c, lw=3.4 if hero else 1.8, alpha=1.0 if hero else 0.75,
                                   z=9 if hero else 7)
            markers[name] = self._torus_marker(c, ms=9 if hero else 6)
            d = np.linalg.norm(tip - P.p0, axis=1)
            self.axplot.semilogy(ts, np.maximum(d, 1e-6), color=c, lw=2.4 if hero else 1.3, ls=ls)
            L = np.sum(np.linalg.norm(np.diff(tip, axis=0), axis=1))
            chord = max(np.linalg.norm(tip[-1] - tip[0]), 1e-9)
            end = pm.wrap_angle(p[-1])
            reached = np.argmin(np.linalg.norm(pm.wrap_angle(end - mins), axis=1))
            elbow = "elbow +" if mins[reached][1] > 0 else "elbow −"
            done = d < 0.1 * d[0]
            t90 = ts[np.argmax(done)] if done.any() else np.nan
            stats.append(f"  {name:9s} {elbow}   {L / chord:4.2f}         {t90:4.1f}")
        self.axflat.plot(*self.q_start, "o", color="white", mec="black", ms=8, zorder=10)
        self.axws.legend(fontsize=8, loc="lower left")
        self.axplot.set_ylim(1e-4, 5)
        self._plot_labels("Same V, four metrics", "hand distance to target")
        D = self.metrics["damping"]
        self.status.set_text(
            f"Start ({self.q_start[0]:+.2f}, {self.q_start[1]:+.2f}) — click the chart to move\n"
            "  metric    reaches   hand path/chord  t to 90%\n"
            + "\n".join(stats)
            + f"\n  D_q={D.d_q:.2f}  D_p=diag({D.D_p[0,0]:.2f},{D.D_p[1,1]:.2f})  D_r={D.d_r:.2f}")

        frames = subsample(len(ts))
        cur = self._cursor()

        def step(k):
            n = frames[k]
            for name in METRIC_NAMES:
                flat_lines[name].set_data(*flat_xy(paths[name][: n + 1]))
                line, tip = hands[name]
                line.set_data(tip[: n + 1, 0], tip[: n + 1, 1])
                arms[name].set_data(*arm_xy(paths[name][n]))
                self._place(markers[name], paths[name][n])
            cur.set_xdata([ts[n], ts[n]])
            return n
        self.n_frames, self.step = len(frames), step

    # ------------------------------------------------------------- scene 3
    def _rate_background(self, extra=""):
        lam = self.rate_grid()
        # Each side scaled on its own: the separating rates are ~10x larger than the
        # converging ones, and a symmetric scale left the green region pale yellow.
        pos, neg = lam[lam > 0], -lam[lam < 0]
        hi = max(np.percentile(pos, 92), 1e-3) if pos.size else 1.0
        lo = max(np.percentile(neg, 92), 1e-3) if neg.size else 1.0
        norm = TwoSlopeNorm(vmin=-lo, vcenter=0.0, vmax=hi)
        cmap = plt.cm.RdYlGn
        self._surface(lam, cmap, norm)
        self._image(lam, cmap, norm)
        self.axflat.contour(self.grid, self.grid, lam.T, levels=[0.0], colors="black", linewidths=1.3)
        which = "identity metric (from Df)" if self.euclid else f"G = {self.metric_name}"
        self.axflat.set_title(f"λ_min certified in {which}{extra}", fontsize=10, color=MUTED)
        return lam

    def _scene_hessian(self):
        P, M = self.potential, self.metric
        lam = self._rate_background()
        self._minimum_markers()
        eps = self.sliders["eps"].val
        ring0 = metric_ring(self.q_start, M.G(self.q_start), eps, n=36)
        t, paths = self._memo(pm.integrate, np.vstack((self.q_start[None], ring0)), P, M,
                              duration=10.0, dt=min(self.dt(), 0.01))
        nominal, ring = paths[:, 0], paths[:, 1:]
        c = METRIC_COLORS[self.metric_name]
        self._flat_path(nominal, c, lw=1.0, alpha=0.35, z=5)
        nom_line = self._dyn(self._flat_path(nominal[:1], c, lw=2.6, z=8, halo=True))
        self._torus_path(nominal, c, lw=0.9, alpha=0.35, halo=False, z=8)
        self._torus_path(nominal, c, lw=2.6, z=11, reveal=True)
        # ring members that end at another minimum
        split = np.linalg.norm(pm.wrap_angle(ring[-1] - nominal[-1]), axis=1) > 0.3
        for j in np.nonzero(split)[0]:
            self._flat_path(ring[:, j], "white", lw=1.0, alpha=0.8, z=6)

        delta = pm.wrap_angle(ring - nominal[:, None])
        Gn = M.G(nominal)
        if self.euclid:
            A = pm.flow_jacobian(nominal, P, M)
            S = -0.5 * (A + np.swapaxes(A, -1, -2))
            lo, hi = pm.generalized_eigs(S, np.broadcast_to(np.eye(2), S.shape))
            d = np.linalg.norm(delta, axis=-1)
            eps0 = d[0].max()
        else:
            H, _ = pm.covariant_hessian(nominal, P, M)
            lo, hi = pm.generalized_eigs(H, Gn)
            d = np.sqrt(np.einsum("tni,tij,tnj->tn", delta, Gn, delta))
            eps0 = eps
        dt = np.diff(t)
        env_lo = eps0 * np.exp(-cumulative_rate(lo, dt))
        env_hi = eps0 * np.exp(-cumulative_rate(hi, dt))
        for j in range(d.shape[1]):
            self.axplot.semilogy(t, d[:, j], color=c, lw=0.8, alpha=0.55)
        self.axplot.fill_between(t, env_hi, env_lo, color=FG, alpha=0.10,
                                 label="Hessian band  λ_max … λ_min")
        self.axplot.semilogy(t, env_lo, "--", color=FG, lw=1.3,
                             label=r"linearised  $\epsilon\,e^{-\int\lambda_{min}dt}$")
        self.axplot.set_ylim(max(np.min(d[d > 0]) * 0.3, 1e-9), max(d.max(), env_lo.max()) * 4)
        self.axplot.legend(fontsize=7.5, loc="lower left")
        self._plot_labels("Perturbation response around the nominal motion",
                          "|δq| to nominal (identity)" if self.euclid else "|δq|_G to nominal")

        self._target()
        grow = d.max() / eps0
        self.status.set_text(
            f"Ring of {ring.shape[1]} starts at G-distance ε = {eps:.2f}\n"
            f"  around ({self.q_start[0]:+.2f}, {self.q_start[1]:+.2f}) — click the chart to move\n"
            f"  at the start: λ_min = {lo[0]:+.2f}, λ_max = {hi[0]:+.2f}\n"
            f"  largest stretch ×{grow:.1f};  final spread ×{d[-1].max() / eps0:.1e}"
            + (f";  {split.sum()} split off" if split.any() else "") + "\n"
            f"  green: {100 * np.mean(lam > 0):.0f}% of the torus")

        frames = subsample(len(t))
        arm = self._arm(c, lw=3.4, z=9)
        ghost_idx = list(range(0, ring.shape[1], 3))
        ghosts = [self._arm(FG, lw=1.1, alpha=0.4, z=7) for _ in ghost_idx]
        hand_ring = self._dyn(self.axws.plot([], [], color=c, lw=1.8, zorder=8)[0])
        flat_ring = Polygon(ring0, closed=True, facecolor=c, edgecolor="white", lw=1.0,
                            alpha=0.8, zorder=10)
        self.axflat.add_patch(flat_ring)
        self._dyn(flat_ring)
        self.axflat.text(0.015, 0.015, "ring rescaled to constant size: its shape shows Hess_G V",
                         transform=self.axflat.transAxes, fontsize=7.3, color="black",
                         bbox=dict(fc="white", ec="none", alpha=0.6, pad=1.5))
        torus_ring = self._dyn(self.ax3.plot([], [], [], color="white", lw=2.4, zorder=13,
                                             path_effects=HALO)[0])
        mk = self._torus_marker(c)
        cur = self._cursor()

        def step(k):
            n = frames[k]
            nom_line.set_data(*flat_xy(nominal[: n + 1]))
            arm.set_data(*arm_xy(nominal[n]))
            for g, j in zip(ghosts, ghost_idx):
                g.set_data(*arm_xy(ring[n, j]))
            _, tips = pm.forward_kinematics(ring[n])
            hand_ring.set_data(np.r_[tips[:, 0], tips[0, 0]], np.r_[tips[:, 1], tips[0, 1]])
            rel = pm.wrap_angle(ring[n] - nominal[n])
            size = max(np.abs(rel).max(), 1e-12)
            flat_ring.set_xy(pm.wrap_angle(nominal[n]) + rel * 0.38 / size)
            # on the torus the ring is magnified only once it is too small to see
            r = nominal[n] + rel * max(0.3 / size, 1.0)
            torus_ring.set_data_3d(*torus_visible_xyz(np.vstack((r, r[:1])),
                                                      view_vector(*self._view), 0.1))
            self._place(mk, nominal[n])
            cur.set_xdata([t[n], t[n]])
            return n
        self.n_frames, self.step = len(frames), step

    # ------------------------------------------------------------- scene 4
    def _scene_contraction(self):
        M = self.metric
        if self.moving:
            P = pm.MovingTarget(radius=self.sliders["radius"].val, omega=0.8,
                                a_q=self.potential.a_q, a_p=self.potential.a_p,
                                a_r=self.potential.a_r, p0=self.potential.p0, k_p=self.potential.k_p)
        else:
            P = self.potential
        self._rate_background("  (target at its mean)" if self.moving else "")
        mins = self._minimum_markers()
        q_min = mins[np.argmin([np.linalg.norm(pm.wrap_angle(m - np.array([1.67, -1.76])))
                                for m in mins])]

        # Omega: largest invariant sublevel set on which the rate stays >= 30% of its value at q*
        V = self.potential.terms(self.Q)[0]
        lam_G = self.rate_grid(euclid=False)
        lam_star = float(pm.contraction_rate(q_min, self.potential, M))
        region, c_lvl, lam_cert = pm.certified_region(self.potential, M, q_min, self.grid, lam_G, V,
                                                      min_rate=0.3 * lam_star)
        if not self.moving:
            self.axflat.contourf(self.grid, self.grid, region.T.astype(float), levels=[0.5, 1.5],
                                 colors=[OMEGA_COLOR], alpha=0.22, zorder=1)
            self.axflat.contour(self.grid, self.grid, region.T.astype(float), levels=[0.5],
                                colors=OMEGA_COLOR, linewidths=2.0, linestyles="--", zorder=2)

        if self.string is None:
            # along the long axis of Omega, from near q* towards its far edge
            pts = self.Q[region]
            axis = pm.wrap_angle(pts[np.argmax(np.linalg.norm(pm.wrap_angle(pts - q_min), axis=1))]
                                 - q_min)
            centre = q_min + 0.45 * axis
            half = 0.4 * np.linalg.norm(axis)
            angle = np.rad2deg(np.arctan2(axis[1], axis[0])) % 180
            self._set_quietly("half", half)
            self._set_quietly("angle", angle)
        else:
            centre, half, angle = self.string
        self._last_string = (np.array(centre, dtype=float), half, angle)
        a = np.deg2rad(angle)
        curve0 = centre + half * np.linspace(-1, 1, 41)[:, None] * np.array([np.cos(a), np.sin(a)])
        dt = min(self.dt(), 0.01)
        r = self._memo(pm.integrate_string, curve0, P, M, duration=10.0, dt=dt,
                       store_every=max(1, int(0.04 / dt)))
        t, curves = r["t"], r["curves"]
        idx = grid_index(curve0, self.n_grid)
        inside = bool(region[idx[:, 0], idx[:, 1]].all())

        c = METRIC_COLORS[self.metric_name]
        _, tracks = self._memo(pm.integrate, np.ascontiguousarray(curve0[::4]), P, M,
                               duration=10.0, dt=dt)
        for j in range(tracks.shape[1]):
            self._flat_path(tracks[:, j], "black", lw=0.8, alpha=0.55, z=6)
            self._torus_path(tracks[:, j], "black", lw=0.8, lift=0.05, z=8, halo=False, alpha=0.6)
        self.axflat.plot(*flat_xy(curve0), color="white", lw=1.3, ls=":", zorder=7)

        if self.euclid:
            L, B, label = r["L_euclid"], r["bound_euclid"], "identity metric"
            self.axplot.semilogy(t, r["L"], color=c, lw=1.0, alpha=0.4, label="length in G")
        else:
            L, B, label = r["L"], r["bound"], f"G = {self.metric_name}"
        self.axplot.semilogy(t, L, color=c, lw=2.8, label=f"string length, {label}")
        self.axplot.semilogy(t, B, "--", color=FG, lw=1.3,
                             label=r"$L_0e^{-\int\lambda_{min}dt}$ along the string")
        if inside and not self.moving and not self.euclid:
            self.axplot.semilogy(t, L[0] * np.exp(-lam_cert * t), color=OMEGA_COLOR, ls=":", lw=2.2,
                                 label=fr"a priori $L_0e^{{-\lambda_\Omega t}}$, $\lambda_\Omega$ = {lam_cert:.2f}")
        self.axplot.set_ylim(max(L.min() * 0.3, 1e-9), max(L.max(), L[0]) * 40)
        self.axplot.legend(fontsize=7.2, loc="lower left")
        self._plot_labels("Length of a curve of starts", "length")

        if self.moving:
            tt = np.linspace(0, 2 * np.pi / 0.8, 120)
            tgt = P.target(tt)
            self.axws.plot(tgt[:, 0], tgt[:, 1], color=TARGET_COLOR, lw=1.2, ls="--", alpha=0.7)
        tgt_mk = self._dyn(self._target())

        ok = bool(np.all(L <= B * 1.01 + 1e-12))
        self.status.set_text(
            f"String of starts: centre ({centre[0]:+.2f}, {centre[1]:+.2f}), L0 = {L[0]:.2f}\n"
            f"  t = {t[-1]:.0f}: L/L0 = {L[-1] / L[0]:.1e};  "
            f"{'bound holds at every step' if ok else 'bound violated'}\n"
            + (f"  Ω = {{V ≤ {c_lvl:.2f}}}, invariant: λ_Ω = {lam_cert:.2f}\n"
               f"  string {'inside Ω → a-priori certificate' if inside else 'leaves Ω: only the bound'}\n"
               if not self.moving else
               "  moving target: no equilibrium, yet every start\n  collapses onto one trajectory\n")
            + ("  certified in the identity metric" if self.euclid else "  click the chart to move the string"))

        stored_t = r["stored_t"]
        arm_main = self._arm(c, lw=3.4, z=9)
        ghosts = [self._arm(FG, lw=1.0, alpha=0.4, z=7) for _ in range(9)]
        str_flat = self._dyn(self.axflat.plot([], [], color=c, lw=3.4, zorder=12, path_effects=HALO)[0])
        str_torus = self._dyn(self.ax3.plot([], [], [], color=c, lw=3.4, zorder=13, path_effects=HALO)[0])
        hand_str = self._dyn(self.axws.plot([], [], color=c, lw=2.4, zorder=8)[0])
        cur = self._cursor()

        def step(k):
            q = curves[k]
            str_flat.set_data(*flat_xy(q))
            # magnify the string on the torus once it has shrunk below visibility
            mid = q[len(q) // 2]
            rel = pm.wrap_angle(q - mid)
            size = np.abs(rel).max()
            qv = mid + rel * (0.12 / size if 0 < size < 0.12 else 1.0)
            str_torus.set_data_3d(*torus_visible_xyz(qv, view_vector(*self._view), 0.1))
            _, tips = pm.forward_kinematics(q)
            hand_str.set_data(tips[:, 0], tips[:, 1])
            arm_main.set_data(*arm_xy(mid))
            for g, j in zip(ghosts, np.linspace(0, len(q) - 1, len(ghosts)).astype(int)):
                g.set_data(*arm_xy(q[j]))
            tg = P.target(stored_t[k])
            tgt_mk.set_data([tg[0]], [tg[1]])
            cur.set_xdata([stored_t[k], stored_t[k]])
            return 0
        self.n_frames, self.step = len(curves), step

    # --------------------------------------------------------------- animation
    def _render_frame(self, k):
        self._last_n = self.step(k)
        self._update_torus_lines(self._last_n)
        self._view_used = self._view

    def _animate(self, _=None):
        self._last_tick = time.time()
        if self._mouse_down and time.time() - self._mouse_seen > 0.2:
            held = left_button_held()
            if held is False:
                self._release("button is up")          # never stay paused on a lost release
            elif held is None and time.time() - self._mouse_seen > 3.0:
                self._release("no release for 3 s")    # platforms without a button query
        if self._busy or self._mouse_down:
            return
        try:
            self._animate_frame()
        except Exception:
            if not self._error_reported:          # report once, keep the window alive
                traceback.print_exc()
                self._error_reported = True

    def _animate_frame(self):
        view = (self.ax3.elev, self.ax3.azim)
        rotated = view != self._view
        if rotated:                               # the user is rotating the torus
            self._view = view
        if self.playing:
            hold = HOLD_FRAMES if self.present else 0
            k = self.frame
            self.frame = (self.frame + 1) % (self.n_frames + hold)
            if k >= self.n_frames and not rotated:
                return                            # holding the last frame
            self._render_frame(min(k, self.n_frames - 1))
        elif rotated or self._view_used != self._view:
            self._update_torus_lines(self._last_n)    # re-mask the hidden side
            self._view_used = self._view
        else:
            return
        if self._bg is None or rotated:
            self.fig.canvas.draw_idle()           # full redraw; _on_draw re-captures the background
        else:
            self.fig.canvas.restore_region(self._bg)
            self._draw_dynamic()

    def export(self, path=None, fps=25):
        os.makedirs("exports", exist_ok=True)
        stem = f"scene{self.scene + 1}_{SCENES[self.scene].split('·')[1].strip().lower()}_{self.metric_name}"
        if self.scene == 3 and self.moving:
            stem += "_moving"
        if FFMpegWriter.isAvailable():
            path = path or os.path.join("exports", stem + ".mp4")
            writer = FFMpegWriter(fps=fps, bitrate=3000)
        else:
            path = path or os.path.join("exports", stem + ".gif")
            writer = PillowWriter(fps=fps)
        was = self.playing
        self.playing = False
        with writer.saving(self.fig, path, dpi=110):
            for k in range(self.n_frames):
                self._render_frame(k)
                writer.grab_frame()
        self.playing = was
        print(f"saved {path}")
        return path

    def snapshot(self, path, frame_fraction=0.35):
        self._render_frame(int(frame_fraction * (self.n_frames - 1)))
        self.fig.savefig(path, dpi=90)


def main():
    if "--snapshot" in sys.argv or "--export-all" in sys.argv:
        matplotlib.use("Agg")
        app = App(interactive=False)
        if "--snapshot" in sys.argv:
            out = sys.argv[sys.argv.index("--snapshot") + 1]
            os.makedirs(out, exist_ok=True)
            shots = [(0, 0.35, False), (1, 0.30, False), (2, 0.12, False), (3, 0.04, False),
                     (3, 0.3, True)]
            for scene, frac, moving in shots:
                if moving != app.moving:
                    app.c_opts.set_active(1)
                app.r_scene.set_active(scene)
                name = f"scene{scene + 1}{'_moving' if moving else ''}.png"
                app.snapshot(os.path.join(out, name), frame_fraction=frac)
        else:
            for scene in range(4):
                app.r_scene.set_active(scene)
                app.export()
            app.c_opts.set_active(1)
            app.export()
        return
    App(present="--present" in sys.argv, debug="--debug" in sys.argv)
    plt.show()


if __name__ == "__main__":
    main()
