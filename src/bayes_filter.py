"""Variational Bayes filter and EFE planner for the quadratic observation model

    y_k = θ₁ x_k² + θ₂ x_k + θ₃ + ε_y       ε_y ~ N(0, σ(l_k))

Derivation lives in `NONLINEAR_MATH.md`. Key points:
  • φ(x) = [x², x, 1]ᵀ. q(θ) closed-form Gaussian (3-dim).
  • q(x) — log-posterior is quartic; we use a second-order Taylor /
    Newton-step Gaussian approximation around the current iterate.
  • Planning predictive variance uses second-order Taylor on h(x),
    which is exact for our quadratic h, plus the full Gaussian
    moments of φ(x)ᵀ V_θ φ(x).
"""
import numpy as np


# ---------------------------------------------------------------------------
# Feature map and Gaussian-moment helpers.
# ---------------------------------------------------------------------------

def phi(x):
    """φ(x) = [x², x, 1]ᵀ."""
    return np.array([x*x, x, 1.0])


def _gaussian_moments(mu, v):
    """Central + raw moments of x ~ N(mu, v) up to 4th order."""
    return {
        "Ex":  mu,
        "Ex2": mu*mu + v,
        "Ex3": mu**3 + 3*mu*v,
        "Ex4": mu**4 + 6*mu*mu*v + 3*v*v,
    }


def _E_phi(mu, v):
    m = _gaussian_moments(mu, v)
    return np.array([m["Ex2"], m["Ex"], 1.0])


def _E_phi_phiT(mu, v):
    """E[φ(x)φ(x)ᵀ] for x ~ N(mu, v).  3x3 matrix."""
    m = _gaussian_moments(mu, v)
    return np.array([
        [m["Ex4"], m["Ex3"], m["Ex2"]],
        [m["Ex3"], m["Ex2"], m["Ex"]],
        [m["Ex2"], m["Ex"],  1.0    ],
    ])


def _E_phi_phiT_cross(mi, vi, mj, vj, c):
    """E[φ(x_i)φ(x_j)ᵀ] for jointly Gaussian (x_i, x_j) with means (mi, mj),
    variances (vi, vj) and covariance c.  3x3 matrix of cross-moments
    E[x_i^a x_j^b], a,b ∈ {2,1,0}.  Used by the full-covariance objective to
    fill the off-diagonal of the measurement block when the rolled-forward
    states are correlated (random walk).  Reduces to _E_phi_phiT when
    (mi,vi)==(mj,vj) and c=vi (the i==j case)."""
    Exi2xj2 = mi*mi*mj*mj + mi*mi*vj + mj*mj*vi + 4*mi*mj*c + vi*vj + 2*c*c
    Exi2xj  = mi*mi*mj + 2*mi*c + mj*vi
    Exixj2  = mi*mj*mj + 2*mj*c + mi*vj
    Exixj   = mi*mj + c
    return np.array([
        [Exi2xj2, Exi2xj,    mi*mi + vi],
        [Exixj2,  Exixj,     mi        ],
        [mj*mj + vj, mj,     1.0       ],
    ])


# ---------------------------------------------------------------------------
# Filter / planner.
# ---------------------------------------------------------------------------

class BayesFilter:
    def __init__(self, delta, sigma, xi, sigma_t, zeta_x, zeta_t, travel_cost,
                 mtheta0, Vtheta0, m_x0, v_x0, m_t0, v_t0, a_lim, b_lim):
        self.delta = delta
        self.sigma = sigma
        self.xi = xi
        self.sigma_t = sigma_t
        self.zeta_x = zeta_x
        self.zeta_t = zeta_t
        self.travel_cost = travel_cost
        self.a_lim = a_lim
        self.b_lim = b_lim

        self.m_theta = np.asarray(mtheta0, dtype=float).copy()
        self.V_theta = np.asarray(Vtheta0, dtype=float).copy()
        self.m_x = float(m_x0)
        self.v_x = float(v_x0)
        self.m_t = float(m_t0)
        self.v_t = float(v_t0)
        self.l = 0

    # -----------------------------------------------------------------------
    # CAVI factors.
    # -----------------------------------------------------------------------

    def q_theta(self, l, y, m_x, v_x, m_theta_prior, V_theta_prior):
        """Closed-form Gaussian update for q(θ) — see math §2."""
        E_phi      = _E_phi(m_x, v_x)
        E_phi_phiT = _E_phi_phiT(m_x, v_x)

        precision_prior = np.linalg.inv(V_theta_prior)
        xi_theta = precision_prior @ m_theta_prior + y * E_phi / self.sigma[l]
        Lambda   = precision_prior + E_phi_phiT / self.sigma[l]

        V_theta_new = np.linalg.inv(Lambda)
        m_theta_new = V_theta_new @ xi_theta
        return m_theta_new, V_theta_new

    def q_x(self, l, y, u, m_x_prior, v_x_prior, m_theta, V_theta, x0=None,
            tol=1e-8, max_iter=20):
        """Laplace approximation of q(x): iterate Newton to the mode of f(x)
        at fixed (m_θ, V_θ), then read curvature at the mode.

        f(x) = -log q̃(x) has cubic score f'(x) and quadratic curvature f''(x);
        see paper appendix eqs. (151–153) and `NONLINEAR_MATH.md §3`.
        Newton: x ← x − f'(x)/f''(x), initialised at x0 (defaults to the
        predicted post-action mean).  Converges in a handful of steps because
        the prior contribution 1/v_pred to f'' is strictly positive.
        """
        if x0 is None:
            x0 = m_x_prior + u

        m_pred = m_x_prior + u
        v_pred = v_x_prior + self.zeta_x
        sig = self.sigma[l]
        a, b, c = m_theta[0], m_theta[1], m_theta[2]
        V11, V22 = V_theta[0, 0], V_theta[1, 1]
        V12, V13, V23 = V_theta[0, 1], V_theta[0, 2], V_theta[1, 2]
        inv_v_pred = 1.0 / v_pred

        x = x0
        neg_Lpp = inv_v_pred  # fallback if loop doesn't execute
        for _ in range(max_iter):
            # Likelihood derivatives at x.
            hp  = 2.0 * a * x + b              # h'(x)
            hpp = 2.0 * a                      # h''(x) — constant in x
            r   = y - (a * x * x + b * x + c)  # residual

            # ½ Q'(x) and ½ Q''(x).
            half_Qp  = (2.0*V11*x**3
                        + 3.0*V12*x*x
                        + (V22 + 2.0*V13)*x
                        + V23)
            half_Qpp = (6.0*V11*x*x
                        + 6.0*V12*x
                        + (V22 + 2.0*V13))

            Lp      = -(x - m_pred) * inv_v_pred + r * hp / sig - half_Qp / sig
            neg_Lpp = inv_v_pred + (hp*hp - r*hpp) / sig + half_Qpp / sig

            # Curvature guard: fall back to the prior precision if the
            # quadratic approximation is non-convex at this iterate.
            if neg_Lpp <= 0:
                neg_Lpp = inv_v_pred

            dx = Lp / neg_Lpp
            x = x + dx
            if abs(dx) < tol:
                break

        # Recompute curvature at the converged mode so v_x = 1/f''(x*).
        hp  = 2.0 * a * x + b
        hpp = 2.0 * a
        r   = y - (a * x * x + b * x + c)
        half_Qpp = (6.0*V11*x*x
                    + 6.0*V12*x
                    + (V22 + 2.0*V13))
        neg_Lpp = inv_v_pred + (hp*hp - r*hpp) / sig + half_Qpp / sig
        if neg_Lpp <= 0:
            neg_Lpp = inv_v_pred

        return x, 1.0 / neg_Lpp

    def q_t(self, l, s, u, r, m_t, v_t):
        """Unchanged from the linear branch — q(t) doesn't depend on h."""
        xi_t = ((m_t + self.delta[l] + s*self.xi + self.travel_cost*abs(u))
                / (v_t + self.zeta_t)
                + r / self.sigma_t)
        lam_t = 1.0/(v_t + self.zeta_t) + 1.0/self.sigma_t
        m_t_new = xi_t / lam_t
        v_t_new = 1.0 / lam_t
        return m_t_new, v_t_new

    # -----------------------------------------------------------------------
    # Likelihood / observation helpers.
    # -----------------------------------------------------------------------

    def predicted_y(self, m_theta, m_x):
        """E[y | x = m_x, θ = m_θ] = φ(m_x)ᵀ m_θ."""
        return m_theta @ phi(m_x)

    def nll_observation(self, l, y, m_theta, m_x, sigma):
        m_y = self.predicted_y(m_theta, m_x)
        v_y = sigma[l]
        return 0.5*np.log(2*np.pi*v_y) + 0.5*(y - m_y)**2 / v_y

    # -----------------------------------------------------------------------
    # Planning predictive variance — math §4.4.
    # -----------------------------------------------------------------------

    def _predictive_var_y(self, m_theta, V_theta, m_x, v_x):
        """Var_{q(x)q(θ)}[ φ(x)ᵀ θ ]  for x ~ N(m_x, v_x), θ ~ N(m_θ, V_θ).

        Sum of two terms (math §4.1):
          (★)   E_{q(x)}[ φ(x)ᵀ V_θ φ(x) ]
          (★★)  Var_{q(x)}[ φ(x)ᵀ m_θ ]   (exact since h is quadratic)
        """
        mu, v = m_x, v_x
        a, b, _ = m_theta[0], m_theta[1], m_theta[2]

        # Gaussian moments.
        mu2 = mu*mu
        Ex2 = mu2 + v
        Ex3 = mu**3 + 3*mu*v
        Ex4 = mu**4 + 6*mu2*v + 3*v*v

        V11, V22, V33 = V_theta[0, 0], V_theta[1, 1], V_theta[2, 2]
        V12, V13, V23 = V_theta[0, 1], V_theta[0, 2], V_theta[1, 2]

        # (★) E[φᵀ V_θ φ].
        E_phiVphi = (V11*Ex4
                     + 2.0*V12*Ex3
                     + (V22 + 2.0*V13)*Ex2
                     + 2.0*V23*mu
                     + V33)

        # (★★) Var[h(x); m_θ] = v (b + 2 a μ)² + 2 v² a²  (second-order Taylor, exact).
        Var_h = v*(b + 2.0*a*mu)**2 + 2.0*v*v*a*a

        return E_phiVphi + Var_h

    def eig_step(self, l, m_x_pred, v_x_pred, v_t_pred, m_theta, V_theta):
        """Per-step EIG (chain-rule term).  Returns -½ log(1 + N/D)."""
        num = self._predictive_var_y(m_theta, V_theta, m_x_pred, v_x_pred) + v_t_pred
        den = self.sigma[l] + self.sigma_t
        return 0.5 * np.log(den / (num + den))

    # -----------------------------------------------------------------------
    # Filter step — CAVI iteration.
    # -----------------------------------------------------------------------

    def _cavi(self, l, y, u, m_theta_prior, V_theta_prior, m_x_prior, v_x_prior,
              n_iter=5):
        """Run CAVI for q(θ)q(x).  Returns (m_theta, V_theta, m_x, v_x)."""
        # Initialise q(x) at the predicted post-action position (Newton step
        # would otherwise see a stale m_x and could diverge under tight σ(l)).
        m_x = m_x_prior + u
        v_x = v_x_prior + self.zeta_x

        m_theta, V_theta = m_theta_prior, V_theta_prior
        for _ in range(n_iter):
            m_theta, V_theta = self.q_theta(l, y, m_x, v_x,
                                            m_theta_prior, V_theta_prior)
            m_x, v_x = self.q_x(l, y, u, m_x_prior, v_x_prior,
                                m_theta, V_theta, x0=m_x)
        return m_theta, V_theta, m_x, v_x

    def step(self, l, y, r, s, u):
        m_theta, V_theta, m_x, v_x = self._cavi(
            l, y, u,
            self.m_theta, self.V_theta, self.m_x, self.v_x,
        )
        m_t, v_t = self.q_t(l, s, u, r, self.m_t, self.v_t)

        self.m_theta = m_theta
        self.V_theta = V_theta
        self.m_x = m_x
        self.v_x = v_x
        self.m_t = m_t
        self.v_t = v_t
        self.l = l
        return self.nll_observation(l, y, m_theta, m_x, self.sigma)

    # -----------------------------------------------------------------------
    # Multistep EFE planner — same enumeration tree, new predictive variance.
    # -----------------------------------------------------------------------

    def plan_multistep_EFE(self, position_grid, H, T_budget, x, fixed_l=None,
                           beta=0.0, first_step=False, objective="joint",
                           return_G=False, fixed_root_s=None):
        """Depth-H tree search over (s, u) pairs minimising the EFE.

        The EFE is G = -MI + (time goal-prior term).  beta sets the strength of
        a goal prior on time, p̄(r) ∝ exp(-beta·elapsed): its cross-entropy adds
        beta·Σ Δt (the accumulated planned time) to G.  beta=0 is a uniform time
        prior (pure information-seeking).  It is a planning preference only — the
        filter never sees beta.

        objective:
          "joint" — paper's eq. 95 with frozen θ-belief; state rolled forward
                    by deterministic transitions only.
          "chain" — per-step EIG chain rule with belief propagation between
                    planning steps via simulated ŷ.

        Returns (s, u, mi) — root action and the planning MI score.
        """
        m_theta_root = self.m_theta
        V_theta_root = self.V_theta

        def rollout_joint(m_x, v_x, m_t, v_t, l, x_pred, h, num_acc, den_acc, dt_acc):
            if h == 0:
                mi = 0.5 * np.log1p(num_acc / den_acc) if den_acc > 0 else 0.0
                # G = -MI + goal-seeking term.  beta*dt_acc = beta·Σ Δt is the
                # cross-entropy of the time goal prior p̄(r) ∝ exp(-beta·elapsed)
                # (dt_acc telescopes to the total planned elapsed time).
                return (-mi + beta*dt_acc), None, None, mi

            best = None
            is_root = (h == H)
            s_choices = [abs(l - fixed_l)] if fixed_l is not None else [0, 1]
            if is_root and fixed_root_s is not None:
                s_choices = [fixed_root_s]
            for s in s_choices:
                for u in position_grid - x_pred:
                    l_next = abs(l - s)
                    m_x_pred = m_x + u
                    v_x_pred = v_x + self.zeta_x
                    v_t_pred = v_t + self.zeta_t

                    switch_cost = 0.0 if (is_root and first_step) else s*self.xi
                    m_t_pred = m_t + self.delta[l_next] + switch_cost + self.travel_cost*abs(u)

                    if m_t_pred > T_budget:
                        continue

                    x_pred_next = x_pred + u
                    num_i = (self._predictive_var_y(m_theta_root, V_theta_root,
                                                    m_x_pred, v_x_pred)
                             + v_t_pred)
                    den_i = self.sigma[l_next] + self.sigma_t
                    dt_i  = m_t_pred - m_t

                    total, _, _, mi = rollout_joint(
                        m_x_pred, v_x_pred, m_t_pred, v_t_pred, l_next, x_pred_next,
                        h - 1, num_acc + num_i, den_acc + den_i, dt_acc + dt_i,
                    )
                    if best is None or total < best[0]:
                        best = (total, s, u, mi)
            return best if best is not None else (float("inf"), 0, 0.0, 0.0)

        def rollout_joint_det(m_x, v_x, m_t, v_t, l, x_pred, h, recs, dt_acc):
            """As rollout_joint, but the MI is the full multivariate Gaussian
            bound ½·log(det Cov_pred / det Cov_lik) instead of the scalar
            ½·log(1 + ΣN/ΣD).

            The determinant does not decompose into a per-step sum, so we defer
            scoring to the leaf: each branch appends a per-step record
            (μ^{x_i}, Var[h_i], v_{t_i}, σ(l_i)) and the H×H covariance blocks
            are assembled once the horizon is exhausted.  The measurement block
            Σ_h carries the exact per-step variance on its diagonal and the
            shared-θ coupling (μ^{x_i})ᵀ V_θ μ^{x_j} off-diagonal; the time block
            is kept diagonal (marginal v_{t_i}), matching the scalar objective.
            """
            if h == 0:
                if not recs:
                    return (beta*dt_acc), None, None, 0.0
                M = np.array([rec[0] for rec in recs])            # (H, 3) feature means
                Sigma_h = M @ V_theta_root @ M.T                  # shared-θ coupling
                np.fill_diagonal(Sigma_h, [rec[1] for rec in recs])  # exact per-step Var[h]
                D_y = np.diag([rec[3] for rec in recs])           # likelihood noise σ(l_i)
                _, ld_num = np.linalg.slogdet(D_y + Sigma_h)
                _, ld_den = np.linalg.slogdet(D_y)
                meas = 0.5 * (ld_num - ld_den)                    # ½ logdet(I + D_y⁻¹Σ_h)
                time = 0.5 * sum(np.log1p(rec[2] / self.sigma_t) for rec in recs)
                mi = meas + time
                return (-mi + beta*dt_acc), None, None, mi

            best = None
            is_root = (h == H)
            s_choices = [abs(l - fixed_l)] if fixed_l is not None else [0, 1]
            if is_root and fixed_root_s is not None:
                s_choices = [fixed_root_s]
            for s in s_choices:
                for u in position_grid - x_pred:
                    l_next = abs(l - s)
                    m_x_pred = m_x + u
                    v_x_pred = v_x + self.zeta_x
                    v_t_pred = v_t + self.zeta_t

                    switch_cost = 0.0 if (is_root and first_step) else s*self.xi
                    m_t_pred = m_t + self.delta[l_next] + switch_cost + self.travel_cost*abs(u)

                    if m_t_pred > T_budget:
                        continue

                    x_pred_next = x_pred + u
                    rec = (
                        _E_phi(m_x_pred, v_x_pred),
                        self._predictive_var_y(m_theta_root, V_theta_root,
                                               m_x_pred, v_x_pred),
                        v_t_pred,
                        self.sigma[l_next],
                    )
                    dt_i = m_t_pred - m_t

                    total, _, _, mi = rollout_joint_det(
                        m_x_pred, v_x_pred, m_t_pred, v_t_pred, l_next, x_pred_next,
                        h - 1, recs + [rec], dt_acc + dt_i,
                    )
                    if best is None or total < best[0]:
                        best = (total, s, u, mi)
            return best if best is not None else (float("inf"), 0, 0.0, 0.0)

        def rollout_joint_det_full(m_x, v_x, m_t, v_t, l, x_pred, h, recs, dt_acc):
            """As rollout_joint_det, but uses the FULL posterior-predictive
            covariance rather than the shared-θ-only approximation:

              * measurement block Σ_h off-diagonals use the full bivariate
                cross-moments E[φ(x_i)φ(x_j)ᵀ], i.e. they also carry the state
                random-walk correlation Cov(x_i,x_j)=v_{x,min(i,j)} (not just the
                shared-θ coupling);
              * time block Σ_τ is the Brownian covariance Cov(t_i,t_j)=
                v_{t,min(i,j)} instead of a diagonal of marginal v_{t_i}.

            Records per step are (m_x_i, v_x_i, v_t_i, σ(l_i)).
            """
            if h == 0:
                if not recs:
                    return (beta*dt_acc), None, None, 0.0
                m_xs = [rec[0] for rec in recs]
                v_xs = [rec[1] for rec in recs]
                v_ts = [rec[2] for rec in recs]
                n = len(recs)
                mu = [_E_phi(m_xs[i], v_xs[i]) for i in range(n)]
                Sigma_h = np.empty((n, n))
                for i in range(n):
                    for j in range(i, n):
                        c = min(v_xs[i], v_xs[j])          # random-walk Cov(x_i,x_j)
                        M = _E_phi_phiT_cross(m_xs[i], v_xs[i], m_xs[j], v_xs[j], c)
                        val = (m_theta_root @ (M - np.outer(mu[i], mu[j])) @ m_theta_root
                               + np.trace(V_theta_root @ M))
                        Sigma_h[i, j] = Sigma_h[j, i] = val
                Sigma_t = np.array([[min(v_ts[i], v_ts[j]) for j in range(n)]
                                    for i in range(n)])     # Brownian time cov
                D_y = np.diag([rec[3] for rec in recs])
                D_t = self.sigma_t * np.eye(n)
                _, ldn_y = np.linalg.slogdet(D_y + Sigma_h)
                _, ldd_y = np.linalg.slogdet(D_y)
                _, ldn_t = np.linalg.slogdet(D_t + Sigma_t)
                _, ldd_t = np.linalg.slogdet(D_t)
                mi = 0.5*(ldn_y - ldd_y) + 0.5*(ldn_t - ldd_t)
                return (-mi + beta*dt_acc), None, None, mi

            best = None
            is_root = (h == H)
            s_choices = [abs(l - fixed_l)] if fixed_l is not None else [0, 1]
            if is_root and fixed_root_s is not None:
                s_choices = [fixed_root_s]
            for s in s_choices:
                for u in position_grid - x_pred:
                    l_next = abs(l - s)
                    m_x_pred = m_x + u
                    v_x_pred = v_x + self.zeta_x
                    v_t_pred = v_t + self.zeta_t

                    switch_cost = 0.0 if (is_root and first_step) else s*self.xi
                    m_t_pred = m_t + self.delta[l_next] + switch_cost + self.travel_cost*abs(u)

                    if m_t_pred > T_budget:
                        continue

                    x_pred_next = x_pred + u
                    rec = (m_x_pred, v_x_pred, v_t_pred, self.sigma[l_next])
                    dt_i = m_t_pred - m_t

                    total, _, _, mi = rollout_joint_det_full(
                        m_x_pred, v_x_pred, m_t_pred, v_t_pred, l_next, x_pred_next,
                        h - 1, recs + [rec], dt_acc + dt_i,
                    )
                    if best is None or total < best[0]:
                        best = (total, s, u, mi)
            return best if best is not None else (float("inf"), 0, 0.0, 0.0)

        def rollout_joint_prop(m_x, v_x, m_t, v_t, l, x_pred, h,
                               Lambda, num_acc, den_acc, dt_acc):
            """As rollout_joint, but PROPAGATE the θ-precision down the rollout.

            Because the model is linear in θ, a measurement's contribution to
            q(θ)'s precision, E[φφᵀ]/σ, is independent of the observed value —
            so V_θ can be shrunk deterministically during planning (no ŷ, no
            CAVI).  This restores the diminishing-returns signal the frozen
            rollout_joint lacks.  m_θ stays frozen (the mean needs data).
            """
            if h == 0:
                mi = 0.5 * np.log1p(num_acc / den_acc) if den_acc > 0 else 0.0
                return (-mi + beta*dt_acc), None, None, mi

            best = None
            is_root = (h == H)
            s_choices = [abs(l - fixed_l)] if fixed_l is not None else [0, 1]
            if is_root and fixed_root_s is not None:
                s_choices = [fixed_root_s]
            for s in s_choices:
                for u in position_grid - x_pred:
                    l_next = abs(l - s)
                    m_x_pred = m_x + u
                    v_x_pred = v_x + self.zeta_x
                    v_t_pred = v_t + self.zeta_t

                    switch_cost = 0.0 if (is_root and first_step) else s*self.xi
                    m_t_pred = m_t + self.delta[l_next] + switch_cost + self.travel_cost*abs(u)

                    if m_t_pred > T_budget:
                        continue

                    x_pred_next = x_pred + u
                    # belief held FOR this measurement = posterior after the
                    # steps already taken in this plan (incoming Λ).
                    V_theta_step = np.linalg.inv(Lambda)
                    num_i = (self._predictive_var_y(m_theta_root, V_theta_step,
                                                    m_x_pred, v_x_pred)
                             + v_t_pred)
                    den_i = self.sigma[l_next] + self.sigma_t
                    dt_i  = m_t_pred - m_t
                    # data-free precision update carried to the next step.
                    Lambda_next = Lambda + _E_phi_phiT(m_x_pred, v_x_pred) / self.sigma[l_next]

                    total, _, _, mi = rollout_joint_prop(
                        m_x_pred, v_x_pred, m_t_pred, v_t_pred, l_next, x_pred_next,
                        h - 1, Lambda_next,
                        num_acc + num_i, den_acc + den_i, dt_acc + dt_i,
                    )
                    if best is None or total < best[0]:
                        best = (total, s, u, mi)
            return best if best is not None else (float("inf"), 0, 0.0, 0.0)

        def rollout_chain(m_theta, V_theta, m_x, v_x, m_t, v_t, l, x_pred, h):
            if h == 0:
                return 0.0, None, None, 0.0

            best = None
            is_root = (h == H)
            s_choices = [abs(l - fixed_l)] if fixed_l is not None else [0, 1]
            if is_root and fixed_root_s is not None:
                s_choices = [fixed_root_s]
            for s in s_choices:
                for u in position_grid - x_pred:
                    l_next = abs(l - s)
                    m_x_pred = m_x + u
                    v_x_pred = v_x + self.zeta_x
                    v_t_pred = v_t + self.zeta_t

                    switch_cost = 0.0 if (is_root and first_step) else s*self.xi
                    m_t_pred = m_t + self.delta[l_next] + switch_cost + self.travel_cost*abs(u)

                    if m_t_pred > T_budget:
                        continue

                    x_pred_next = x_pred + u

                    g = self.eig_step(l_next, m_x_pred, v_x_pred, v_t_pred,
                                      m_theta, V_theta)
                    # g = eig_step = -EIG (per-step -MI), so the time goal-prior
                    # term beta·Δt is ADDED: G_step = -EIG + beta·Δt.
                    score = g + beta*(m_t_pred - m_t)

                    # Belief propagation through simulated ŷ at the predicted position.
                    y_pred = self.predicted_y(m_theta, m_x_pred)
                    m_theta_n, V_theta_n, m_x_n, v_x_n = self._cavi(
                        l_next, y_pred, u,
                        m_theta, V_theta, m_x, v_x,
                    )
                    m_t_n, v_t_n = self.q_t(l_next, s, u, m_t_pred, m_t, v_t)

                    future, _, _, future_mi = rollout_chain(
                        m_theta_n, V_theta_n, m_x_n, v_x_n,
                        m_t_n, v_t_n, l_next, x_pred_next, h - 1,
                    )
                    total = score + future
                    mi = -g + future_mi

                    if best is None or total < best[0]:
                        best = (total, s, u, mi)
            return best if best is not None else (float("inf"), 0, 0.0, 0.0)

        if objective == "joint":
            G, s, u, mi = rollout_joint(
                self.m_x, self.v_x, self.m_t, self.v_t, self.l, x, H,
                num_acc=0.0, den_acc=0.0, dt_acc=0.0,
            )
        elif objective == "joint_det":
            G, s, u, mi = rollout_joint_det(
                self.m_x, self.v_x, self.m_t, self.v_t, self.l, x, H,
                recs=[], dt_acc=0.0,
            )
        elif objective == "joint_det_full":
            G, s, u, mi = rollout_joint_det_full(
                self.m_x, self.v_x, self.m_t, self.v_t, self.l, x, H,
                recs=[], dt_acc=0.0,
            )
        elif objective == "joint_prop":
            G, s, u, mi = rollout_joint_prop(
                self.m_x, self.v_x, self.m_t, self.v_t, self.l, x, H,
                np.linalg.inv(V_theta_root),
                num_acc=0.0, den_acc=0.0, dt_acc=0.0,
            )
        elif objective == "chain":
            G, s, u, mi = rollout_chain(
                self.m_theta, self.V_theta, self.m_x, self.v_x,
                self.m_t, self.v_t, self.l, x, H,
            )
        else:
            raise ValueError(f"unknown objective: {objective!r}")
        if return_G:
            return s, u, mi, G
        return s, u, mi
