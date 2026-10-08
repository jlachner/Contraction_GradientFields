"""Checks of the geometry behind the scenes.  Run:  python test_pendulum_gradient_model.py
(also collected by pytest)."""

import numpy as np

import pendulum_gradient_model as pm

RNG = np.random.default_rng(7)
Q = RNG.uniform(-3.0, 3.0, (400, 2))
EPS = 1e-6
E = np.eye(2)
POTENTIAL = pm.Potential(a_q=0.4, a_p=1.0, a_r=0.5, phi0=0.7, p0=(0.9, -0.9))
METRICS = dict(pm.METRICS, damping_aniso=pm.Damping(d_q=0.2, D_p=[[1.5, 0.3], [0.3, 0.6]], d_r=0.4))


def central(fn, q):
    return np.stack([(fn(q + EPS * E[k]) - fn(q - EPS * E[k])) / (2 * EPS) for k in range(2)], axis=-1)


def test_potential_derivatives():
    V, dV, ddV = POTENTIAL.terms(Q)
    assert np.allclose(central(lambda x: POTENTIAL.terms(x)[0], Q), dV, atol=1e-7)
    assert np.allclose(central(lambda x: POTENTIAL.terms(x)[1], Q), ddV, atol=1e-7)
    assert np.allclose(POTENTIAL.grad(Q), dV, atol=1e-12)
    moving = pm.MovingTarget(a_q=0.3)
    assert np.allclose(moving.grad(Q, 1.3), moving.terms(Q, 1.3)[1], atol=1e-12)


def test_metric_derivatives_and_fast_path():
    for name, M in METRICS.items():
        G, dG = M.G_dG(Q)
        fd = np.moveaxis(central(M.G, Q), -1, -3)
        assert np.allclose(fd, dG, atol=1e-7), name
        assert np.allclose(M.G(Q), G, atol=1e-12), name
        assert np.all(np.linalg.eigvalsh(G) > 0), name


def test_damping_connection_closed_form():
    """Gamma^k_ij = G^kl p_l^T D_p p_ij (J_r constant in the plane)."""
    M = METRICS["damping_aniso"]
    G, dG = M.G_dG(Q)
    J, P2 = pm.jacobian_terms(Q)
    closed = np.einsum("nkl,nal,ab,nijb->nkij", pm.inv2(G), J, M.D_p, P2)
    assert np.allclose(closed, pm.christoffel(G, dG), atol=1e-12)


def test_flow_jacobian():
    for name, M in METRICS.items():
        fd = central(lambda x: pm.flow(x, POTENTIAL, M), Q)
        assert np.allclose(fd, pm.flow_jacobian(Q, POTENTIAL, M), atol=1e-6), name


def test_contraction_identity():
    """(Df)^T G + G Df + Gdot = -2 Hess_G V for every metric."""
    for name, M in METRICS.items():
        G, dG = M.G_dG(Q)
        A = pm.flow_jacobian(Q, POTENTIAL, M)
        Gdot = np.einsum("nkij,nk->nij", dG, pm.flow(Q, POTENTIAL, M))
        lhs = np.swapaxes(A, -1, -2) @ G + G @ A + Gdot
        H, _ = pm.covariant_hessian(Q, POTENTIAL, M)
        assert np.allclose(lhs, -2 * H, atol=1e-10), name


def test_minima_are_metric_independent():
    P = pm.Potential(p0=(0.9, -0.9))
    ref = pm.find_minima(P, pm.Identity())
    for name, M in pm.METRICS.items():
        mins = pm.find_minima(P, M, n=12)
        assert len(mins) == len(ref), name
        for m in mins:
            assert np.min(np.linalg.norm(pm.wrap_angle(ref - m), axis=1)) < 0.05, name


def test_conformal_metric_keeps_paths():
    P = pm.Potential(p0=(0.9, -0.9))
    _, a = pm.integrate(np.array([-1.2, -0.4]), P, pm.Identity(), duration=30, dt=0.005)
    _, b = pm.integrate(np.array([-1.2, -0.4]), P, pm.Conformal(), duration=60, dt=0.005)
    # every point of the conformal path lies on the identity path
    d = np.linalg.norm(b[::50, None, :] - a[None, :, :], axis=-1).min(axis=1)
    assert d.max() < 1e-2


def test_string_bound_holds():
    """L(t) <= L0 exp(-int lambda_min) for strings inside, across and beyond the ridge."""
    strings = [((1.67 + 0.5, -1.76 + 0.4), 0.45, (1, -1)),
               ((0.3, -1.0), 0.9, (1, 0.6)),
               ((0.8, 0.0), 0.5, (1, -1))]
    for moving in (False, True):
        P = pm.MovingTarget(radius=0.35, omega=0.8, p0=(0.9, -0.9)) if moving else pm.Potential(p0=(0.9, -0.9))
        for centre, half, d in strings:
            d = np.asarray(d, float) / np.linalg.norm(d)
            curve = np.asarray(centre) + half * np.linspace(-1, 1, 41)[:, None] * d
            for name in ("identity", "damping", "inertia"):
                r = pm.integrate_string(curve, P, pm.METRICS[name], duration=8.0, dt=0.01)
                assert np.all(r["L"] <= r["bound"] * 1.01 + 1e-12), (moving, centre, name)
                assert np.all(r["L_euclid"] <= r["bound_euclid"] * 1.01 + 1e-12), (moving, centre, name)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} checks passed")
