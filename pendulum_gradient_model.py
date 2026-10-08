"""Natural-gradient flow of a planar two-link arm (double pendulum) on its torus.

Everything is analytic and vectorised: every function accepts configurations of
shape (..., 2) and returns arrays with matching leading dimensions.

    flow             qdot = -M(q)^-1 dV(q)
    covariant Hess   Hess_M V = d2V - Gamma(M)^k dV_k
    contraction      (Df)^T M + M Df + Mdot = -2 Hess_M V
                     => rate = generalised eigenvalue  Hess_M V v = lambda M v

Angles: theta1 is measured from the downward vertical, theta2 is the elbow angle
relative to link 1, so the arm hangs straight down at q = (0, 0).
"""

from __future__ import annotations

import numpy as np

TAU = 2.0 * np.pi
L1, L2 = 1.0, 1.0


def wrap_angle(x):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(x) + np.pi) % TAU - np.pi


# --------------------------------------------------------------------------
# Kinematics
# --------------------------------------------------------------------------

def forward_kinematics(q):
    """Elbow and end-effector positions, each (..., 2)."""
    t1, t12 = q[..., 0], q[..., 0] + q[..., 1]
    elbow = np.stack((L1 * np.sin(t1), -L1 * np.cos(t1)), axis=-1)
    tip = elbow + np.stack((L2 * np.sin(t12), -L2 * np.cos(t12)), axis=-1)
    return elbow, tip


def jacobian_terms(q):
    """End-effector Jacobian columns p_i and second derivatives p_ij.

    Returns J (..., 2, 2) with J[..., :, i] = dp/dq_i and
    P2 (..., 2, 2, 2) with P2[..., i, j, :] = d2p/dq_i dq_j.
    """
    t1, t12 = q[..., 0], q[..., 0] + q[..., 1]
    c1, s1, c12, s12 = np.cos(t1), np.sin(t1), np.cos(t12), np.sin(t12)
    J = np.empty(q.shape[:-1] + (2, 2))
    J[..., 0, 0] = L1 * c1 + L2 * c12
    J[..., 1, 0] = L1 * s1 + L2 * s12
    J[..., 0, 1] = L2 * c12
    J[..., 1, 1] = L2 * s12
    p11 = np.stack((-L1 * s1 - L2 * s12, L1 * c1 + L2 * c12), axis=-1)
    p12 = np.stack((-L2 * s12, L2 * c12), axis=-1)
    P2 = np.empty(q.shape[:-1] + (2, 2, 2))
    P2[..., 0, 0, :] = p11
    P2[..., 0, 1, :] = p12
    P2[..., 1, 0, :] = p12
    P2[..., 1, 1, :] = p12
    return J, P2


# --------------------------------------------------------------------------
# Potentials:  V = a_q V_q + a_p V_p + a_r V_r
# --------------------------------------------------------------------------

class Potential:
    """Joint, end-effector position and end-effector orientation modules."""

    def __init__(self, a_q=0.0, a_p=1.0, a_r=0.0, q0=(0.0, 0.0), k_q=(1.0, 1.0),
                 p0=(0.9, -0.9), k_p=2.0, phi0=0.0, k_r=1.0):
        self.a_q, self.a_p, self.a_r = a_q, a_p, a_r
        self.phi0, self.k_r = phi0, k_r
        self.q0 = np.asarray(q0, dtype=float)
        self.k_q = np.asarray(k_q, dtype=float)
        self.p0 = np.asarray(p0, dtype=float)
        self.k_p = k_p

    def target(self, t=0.0):
        """Cartesian target; overridden by MovingTarget."""
        return self.p0

    def grad(self, q, t=0.0):
        """dV only (fast path for integration)."""
        t1, t12 = q[..., 0], q[..., 0] + q[..., 1]
        c1, s1, c12, s12 = np.cos(t1), np.sin(t1), np.cos(t12), np.sin(t12)
        p0 = self.target(t)
        ex = L1 * s1 + L2 * s12 - p0[..., 0]
        ey = -L1 * c1 - L2 * c12 - p0[..., 1]
        # dV_p = k_p J^T e with J = [[L1 c1 + L2 c12, L2 c12], [L1 s1 + L2 s12, L2 s12]]
        g2 = self.k_p * L2 * (ex * c12 + ey * s12)
        g1 = self.k_p * L1 * (ex * c1 + ey * s1) + g2
        dq = q - self.q0
        dphi = t12 - self.phi0
        r = self.a_r * self.k_r * np.sin(dphi)
        out = np.empty(q.shape)
        out[..., 0] = self.a_p * g1 + self.a_q * self.k_q[0] * np.sin(dq[..., 0]) + r
        out[..., 1] = self.a_p * g2 + self.a_q * self.k_q[1] * np.sin(dq[..., 1]) + r
        return out

    def terms(self, q, t=0.0):
        """Return V (...), dV (..., 2), d2V (..., 2, 2) in coordinates."""
        dq = q - self.q0
        Vq = np.sum(self.k_q * (1.0 - np.cos(dq)), axis=-1)
        dVq = self.k_q * np.sin(dq)
        ddVq = np.zeros(q.shape[:-1] + (2, 2))
        ddVq[..., 0, 0] = self.k_q[0] * np.cos(dq[..., 0])
        ddVq[..., 1, 1] = self.k_q[1] * np.cos(dq[..., 1])

        _, tip = forward_kinematics(q)
        J, P2 = jacobian_terms(q)
        e = tip - self.target(t)
        Vp = 0.5 * self.k_p * np.sum(e * e, axis=-1)
        dVp = self.k_p * np.einsum("...a,...ai->...i", e, J)
        ddVp = self.k_p * (np.einsum("...ai,...aj->...ij", J, J)
                           + np.einsum("...a,...ija->...ij", e, P2))

        # Orientation spring on phi = theta1 + theta2 (J_r = [1, 1]).
        dphi = q[..., 0] + q[..., 1] - self.phi0
        Vr = self.k_r * (1.0 - np.cos(dphi))
        dVr = (self.k_r * np.sin(dphi))[..., None] * np.ones(2)
        ddVr = (self.k_r * np.cos(dphi))[..., None, None] * np.ones((2, 2))

        V = self.a_q * Vq + self.a_p * Vp + self.a_r * Vr
        dV = self.a_q * dVq + self.a_p * dVp + self.a_r * dVr
        ddV = self.a_q * ddVq + self.a_p * ddVp + self.a_r * ddVr
        return V, dV, ddV


class MovingTarget(Potential):
    """Cartesian target that circles around p0."""

    def __init__(self, radius=0.35, omega=0.9, **kw):
        super().__init__(**kw)
        self.radius, self.omega = radius, omega

    def target(self, t=0.0):
        t = np.asarray(t, dtype=float)
        ang = self.omega * t
        return self.p0 + self.radius * np.stack((np.cos(ang), np.sin(ang)), axis=-1)


# --------------------------------------------------------------------------
# Metrics: each returns G (..., 2, 2) and dG (..., 2, 2, 2), dG[..., k, i, j] = d_k G_ij
# --------------------------------------------------------------------------

class Metric:
    name = "metric"

    def G_dG(self, q):
        raise NotImplementedError

    def G(self, q):
        return self.G_dG(q)[0]


class Identity(Metric):
    name = "identity"

    def G(self, q):
        return np.broadcast_to(np.eye(2), q.shape[:-1] + (2, 2)).copy()

    def G_dG(self, q):
        G = np.broadcast_to(np.eye(2), q.shape[:-1] + (2, 2)).copy()
        return G, np.zeros(q.shape[:-1] + (2, 2, 2))


class Conformal(Metric):
    """c(q) I: same direction as the Euclidean gradient, only the speed changes."""
    name = "conformal"

    def __init__(self, gamma=2.5):
        self.gamma = gamma

    def G(self, q):
        c = 1.0 + self.gamma * (1.0 - np.cos(q[..., 1]))
        return c[..., None, None] * np.eye(2)

    def G_dG(self, q):
        c = 1.0 + self.gamma * (1.0 - np.cos(q[..., 1]))
        dc = np.stack((np.zeros_like(c), self.gamma * np.sin(q[..., 1])), axis=-1)
        G = c[..., None, None] * np.eye(2)
        dG = dc[..., :, None, None] * np.eye(2)
        return G, dG


class Damping(Metric):
    """Damping pulled back to the joints: G = D_q + J_p^T D_p J_p + J_r^T D_r J_r.

    From the viscous force balance  G(q) qdot = -dV  this is the natural metric.
    In the plane the end-effector angle is theta1 + theta2, so J_r = [1, 1] is
    constant and only the translational term contributes to dG.  With constant
    symmetric D_p the connection is  Gamma^k_ij = G^kl  p_l^T D_p p_ij.
    """
    name = "damping"

    def __init__(self, d_q=0.15, D_p=((1.0, 0.0), (0.0, 1.0)), d_r=0.0):
        self.d_q, self.d_r = d_q, d_r
        self.D_p = np.asarray(D_p, dtype=float)

    def G(self, q):
        t1, t12 = q[..., 0], q[..., 0] + q[..., 1]
        c1, s1, c12, s12 = np.cos(t1), np.sin(t1), np.cos(t12), np.sin(t12)
        ax, ay = L1 * c1 + L2 * c12, L1 * s1 + L2 * s12      # p_1
        bx, by = L2 * c12, L2 * s12                          # p_2
        (dxx, dxy), (dyx, dyy) = self.D_p

        def form(ux, uy, vx, vy):                            # u^T D_p v
            return ux * (dxx * vx + dxy * vy) + uy * (dyx * vx + dyy * vy)

        G = np.empty(q.shape[:-1] + (2, 2))
        G[..., 0, 0] = self.d_q + form(ax, ay, ax, ay) + self.d_r
        G[..., 1, 1] = self.d_q + form(bx, by, bx, by) + self.d_r
        G[..., 0, 1] = G[..., 1, 0] = form(ax, ay, bx, by) + self.d_r
        return G

    def G_dG(self, q):
        J, P2 = jacobian_terms(q)
        Jr = np.ones(2)
        G = (self.d_q * np.eye(2)
             + np.einsum("...ai,ab,...bj->...ij", J, self.D_p, J)
             + self.d_r * np.outer(Jr, Jr))
        # d_k (p_i^T D_p p_j) = p_ki^T D_p p_j + p_i^T D_p p_kj
        dG = (np.einsum("...kia,ab,...bj->...kij", P2, self.D_p, J)
              + np.einsum("...ai,ab,...kjb->...kij", J, self.D_p, P2))
        return G, dG


class Inertia(Metric):
    """Mass matrix of the double pendulum with point masses at the link ends."""
    name = "inertia"

    def __init__(self, m1=1.0, m2=1.0):
        self.m1, self.m2 = m1, m2

    def G(self, q):
        m1, m2 = self.m1, self.m2
        c2 = np.cos(q[..., 1])
        G = np.empty(q.shape[:-1] + (2, 2))
        G[..., 0, 0] = (m1 + m2) * L1**2 + m2 * L2**2 + 2 * m2 * L1 * L2 * c2
        G[..., 0, 1] = G[..., 1, 0] = m2 * L2**2 + m2 * L1 * L2 * c2
        G[..., 1, 1] = m2 * L2**2
        return G

    def G_dG(self, q):
        m1, m2 = self.m1, self.m2
        c2, s2 = np.cos(q[..., 1]), np.sin(q[..., 1])
        G = np.empty(q.shape[:-1] + (2, 2))
        G[..., 0, 0] = (m1 + m2) * L1**2 + m2 * L2**2 + 2 * m2 * L1 * L2 * c2
        G[..., 0, 1] = G[..., 1, 0] = m2 * L2**2 + m2 * L1 * L2 * c2
        G[..., 1, 1] = m2 * L2**2
        dG = np.zeros(q.shape[:-1] + (2, 2, 2))
        dG[..., 1, 0, 0] = -2 * m2 * L1 * L2 * s2
        dG[..., 1, 0, 1] = dG[..., 1, 1, 0] = -m2 * L1 * L2 * s2
        return G, dG


METRICS = {"identity": Identity(), "conformal": Conformal(),
           "damping": Damping(), "inertia": Inertia()}


# --------------------------------------------------------------------------
# Geometry of the flow
# --------------------------------------------------------------------------

def inv2(G):
    det = G[..., 0, 0] * G[..., 1, 1] - G[..., 0, 1] * G[..., 1, 0]
    inv = np.empty_like(G)
    inv[..., 0, 0] = G[..., 1, 1]
    inv[..., 1, 1] = G[..., 0, 0]
    inv[..., 0, 1] = -G[..., 0, 1]
    inv[..., 1, 0] = -G[..., 1, 0]
    return inv / det[..., None, None]


def christoffel(G, dG):
    """Gamma[..., k, i, j] of the Levi-Civita connection."""
    first = 0.5 * (np.einsum("...ilj->...lij", dG)       # d_i G_lj
                   + np.einsum("...jli->...lij", dG)     # d_j G_li
                   - dG)                                 # d_l G_ij
    return np.einsum("...kl,...lij->...kij", inv2(G), first)


def flow(q, potential, metric, t=0.0):
    """qdot = -G^-1 dV, solved in closed form for 2x2."""
    G = metric.G(q)
    g = potential.grad(q, t)
    a, b, d = G[..., 0, 0], G[..., 0, 1], G[..., 1, 1]
    det = a * d - b * b
    out = np.empty(q.shape)
    out[..., 0] = -(d * g[..., 0] - b * g[..., 1]) / det
    out[..., 1] = -(a * g[..., 1] - b * g[..., 0]) / det
    return out


def covariant_hessian(q, potential, metric, t=0.0):
    G, dG = metric.G_dG(q)
    _, dV, ddV = potential.terms(q, t)
    H = ddV - np.einsum("...kij,...k->...ij", christoffel(G, dG), dV)
    return 0.5 * (H + np.swapaxes(H, -1, -2)), G


def generalized_eigs(H, G):
    """Both eigenvalues of H v = lambda G v for symmetric H and SPD G (2x2)."""
    detG = G[..., 0, 0] * G[..., 1, 1] - G[..., 0, 1] ** 2
    detH = H[..., 0, 0] * H[..., 1, 1] - H[..., 0, 1] ** 2
    B = H[..., 0, 0] * G[..., 1, 1] + H[..., 1, 1] * G[..., 0, 0] - 2 * H[..., 0, 1] * G[..., 0, 1]
    disc = np.sqrt(np.maximum(B * B - 4 * detG * detH, 0.0))
    return (B - disc) / (2 * detG), (B + disc) / (2 * detG)


def contraction_rate(q, potential, metric, t=0.0):
    """lambda_min(M^-1 Hess_M V): positive means neighbours converge at that rate."""
    H, G = covariant_hessian(q, potential, metric, t)
    return generalized_eigs(H, G)[0]


def flow_jacobian(q, potential, metric, t=0.0):
    """Df[..., i, k] = d f_i / d q_k for f = -M^-1 dV."""
    G, dG = metric.G_dG(q)
    _, dV, ddV = potential.terms(q, t)
    Gi = inv2(G)
    term1 = -np.einsum("...ij,...jk->...ik", Gi, ddV)
    # d_k(M^-1) = -M^-1 (d_k M) M^-1, so -d_k(M^-1) dV = M^-1 (d_k M) M^-1 dV
    w = np.einsum("...ij,...j->...i", Gi, dV)
    term2 = np.einsum("...ij,...kjl,...l->...ik", Gi, dG, w)
    return term1 + term2


def euclidean_rate(q, potential, metric, t=0.0):
    """Certificate in the identity metric: lambda_min of -sym(Df)."""
    A = flow_jacobian(q, potential, metric, t)
    S = -0.5 * (A + np.swapaxes(A, -1, -2))
    return generalized_eigs(S, np.broadcast_to(np.eye(2), S.shape))[0]


# --------------------------------------------------------------------------
# Integration and measurement
# --------------------------------------------------------------------------

def integrate(q0, potential, metric, duration=6.0, dt=0.01, t0=0.0):
    """RK4 for any batch of starts q0 (..., 2). Returns times (T,), paths (T, ..., 2)."""
    times = t0 + np.arange(0.0, duration + 0.5 * dt, dt)
    q = np.array(q0, dtype=float)
    out = np.empty((len(times),) + q.shape)
    out[0] = q
    for n in range(1, len(times)):
        t, h = times[n - 1], dt
        k1 = flow(q, potential, metric, t)
        k2 = flow(q + 0.5 * h * k1, potential, metric, t + 0.5 * h)
        k3 = flow(q + 0.5 * h * k2, potential, metric, t + 0.5 * h)
        k4 = flow(q + h * k3, potential, metric, t + h)
        q = q + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[n] = q
    return times, out


def curve_length(curve, metric):
    """Riemannian length of a polyline (..., N, 2) in configuration space."""
    seg = wrap_angle(np.diff(curve, axis=-2))
    mid = curve[..., :-1, :] + 0.5 * seg
    G = metric.G(mid)
    return np.sum(np.sqrt(np.maximum(np.einsum("...i,...ij,...j->...", seg, G, seg), 0.0)), axis=-1)


def find_minima(potential, metric, n=24, duration=40.0, dt=0.02, tol=0.05):
    """Distinct attractors reached from a coarse grid of starts (metric-independent)."""
    g = np.linspace(-np.pi, np.pi, n, endpoint=False) + np.pi / n
    starts = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    _, paths = integrate(starts, potential, metric, duration, dt)
    ends = wrap_angle(paths[-1])
    speed = np.linalg.norm(flow(ends, potential, metric), axis=-1)
    minima = []
    for e in ends[speed < 1e-3]:
        if all(np.linalg.norm(wrap_angle(e - m)) > tol for m in minima):
            minima.append(e)
    return np.array(minima)


def basin_labels(potential, metric, minima, n=90, duration=25.0, dt=0.02, capture=0.12):
    """Index of the minimum reached from each grid start (-1 if none in time).

    Starts are retired once they come within `capture` of a minimum, so the
    cost is set by the slowest starts near the separatrices, not by all of them.
    """
    g = np.linspace(-np.pi, np.pi, n)
    q = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    lab = -np.ones(len(q), dtype=int)
    active = np.arange(len(q))
    for step in range(int(duration / dt)):
        qa = _rk4_step(q[active], 0.0, dt, potential, metric)
        q[active] = qa
        if step % 10 == 9:
            d = np.linalg.norm(wrap_angle(qa[:, None, :] - minima), axis=-1)
            hit = np.min(d, axis=-1) < capture
            lab[active[hit]] = np.argmin(d[hit], axis=-1)
            active = active[~hit]
            if len(active) == 0:
                break
    return g, lab.reshape(n, n)


def _rk4_step(q, t, h, potential, metric):
    k1 = flow(q, potential, metric, t)
    k2 = flow(q + 0.5 * h * k1, potential, metric, t + 0.5 * h)
    k3 = flow(q + 0.5 * h * k2, potential, metric, t + 0.5 * h)
    k4 = flow(q + h * k3, potential, metric, t + h)
    return q + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_string(curve0, potential, metric, duration=10.0, dt=0.01, closed=False,
                     max_seg=0.04, max_points=4000, store_every=4):
    """Flow a curve of initial conditions and track its Riemannian length.

    Segments that stretch beyond `max_seg` (metric length) are refined by
    inserting midpoints, so expanding regions such as a saddle are sampled and
    the rate bound below stays valid where the string splits.

    Returns a dict with times, stored curves (list, variable length), the
    length in M and in the identity metric, and the smallest contraction rate
    along the string in M and in the identity metric at every step.
    """
    eucl = Identity()
    q = np.array(curve0, dtype=float)
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    out = {"t": times, "curves": [], "stored_t": [], "L": np.empty(len(times)),
           "L_euclid": np.empty(len(times)), "lam": np.empty(len(times)),
           "lam_euclid": np.empty(len(times))}

    def closed_view(c):
        return np.vstack((c, c[:1])) if closed else c

    for n, t in enumerate(times):
        # refine where the string has stretched
        for _ in range(3):
            c = closed_view(q)
            seg = wrap_angle(np.diff(c, axis=0))
            G = metric.G(c[:-1] + 0.5 * seg)
            ell = np.sqrt(np.einsum("ni,nij,nj->n", seg, G, seg))
            long = np.nonzero(ell > max_seg)[0]
            if len(long) == 0 or len(q) + len(long) > max_points:
                break
            mids = c[long] + 0.5 * seg[long]
            q = np.insert(q, long + 1, mids, axis=0)
        c = closed_view(q)
        out["L"][n] = curve_length(c, metric)
        out["L_euclid"][n] = curve_length(c, eucl)
        out["lam"][n] = contraction_rate(q, potential, metric, t).min()
        out["lam_euclid"][n] = euclidean_rate(q, potential, metric, t).min()
        if n % store_every == 0:
            out["curves"].append(q.copy())
            out["stored_t"].append(t)
        if n + 1 < len(times):
            q = _rk4_step(q, t, dt, potential, metric)
    out["stored_t"] = np.array(out["stored_t"])
    for key, lam in (("bound", "lam"), ("bound_euclid", "lam_euclid")):
        rate = out[lam]
        integral = np.concatenate(([0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)))
        L0 = out["L"][0] if key == "bound" else out["L_euclid"][0]
        out[key] = L0 * np.exp(-integral)
    return out


def periodic_component(mask, seed):
    """Connected component of a boolean grid on the torus that contains `seed`."""
    comp = np.zeros_like(mask)
    comp[seed] = mask[seed]
    while True:
        grown = (comp | np.roll(comp, 1, 0) | np.roll(comp, -1, 0)
                 | np.roll(comp, 1, 1) | np.roll(comp, -1, 1)) & mask
        if (grown == comp).all():
            return comp
        comp = grown


def certified_region(potential, metric, q_min, grid, lam_grid, V_grid, min_rate=0.0, iters=30):
    """Largest sublevel component {V <= c} around q_min on which lambda > min_rate.

    A sublevel set of V is forward-invariant for any gradient flow, so the
    smallest rate on it is an a-priori exponential certificate for every pair
    of trajectories that starts inside.  Returns (mask, c, lambda_cert).
    """
    seed = tuple(np.argmin(np.abs(wrap_angle(grid[:, None] - q_min)), axis=0))
    lo, hi = V_grid[seed], V_grid.max()
    for _ in range(iters):
        c = 0.5 * (lo + hi)
        if (lam_grid[periodic_component(V_grid <= c, seed)] > min_rate).all():
            lo = c
        else:
            hi = c
    mask = periodic_component(V_grid <= lo, seed)
    return mask, lo, float(lam_grid[mask].min())
