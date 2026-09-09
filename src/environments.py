import numpy as np

# =================================================================================
# Environments
# =================================================================================
class BiRotor:
    def __init__(self, m, l, g, dt, H, Q, R, s0, horizon = 200, num_rollouts = 1):

        self.s_dim = H.shape[1]
        self.o_dim = H.shape[0]
        self.horizon = horizon
        self.num_rollouts = num_rollouts

        self.m = m
        self.l = l
        self.I = 2 * m * l ** 2
        self.g = g

        self.dt = dt

        self.H = H
        self.Q = Q
        self.R = R
        self.proc_noise = np.random.multivariate_normal(np.zeros(self.s_dim), self.Q, [self.num_rollouts, self.horizon]).T
        self.obs_noise = np.random.multivariate_normal(np.zeros(self.o_dim), self.R, [self.num_rollouts, self.horizon]).T

        self.done = False
        self.t = 0
        self.s0 = s0
        self.s = s0

    def step_dot(self, u): # u has shape [dim, num_rollouts]
        x, dx, y, dy, th, dth = self.s
        u1, u2 = u[0], u[1]

        ddx = -(u1 + u2) * np.sin(th) / self.m
        ddy = (u1 + u2) * np.cos(th) / self.m - self.g
        ddth = self.l * (u1 - u2) / self.I

        return np.array([dx, ddx, dy, ddy, dth, ddth])  # increment has shape [dim, num_rollouts]

    def step(self, u):  # u has shape [dim, num_rollouts]
        next_s = self.s + self.dt * self.step_dot(u) + self.proc_noise[:, self.t]
        obs = self.H @ next_s + self.obs_noise[:, self.t]

        self.s = next_s
        self.t += 1

        return next_s, obs

    def reset(self):
        self.proc_noise = np.random.multivariate_normal(np.zeros(self.s_dim), self.Q, [self.num_rollouts, self.horizon]).T
        self.obs_noise = np.random.multivariate_normal(np.zeros(self.o_dim), self.R, [self.num_rollouts, self.horizon]).T

        self.s = self.s0
        obs = self.H @ self.s + self.obs_noise[:, 0]

        self.done = False
        self.t = 0
        return self.s.copy(), obs

class CartPole:
    def __init__(self, m, M, l, g, d, dt, H, Q, R, s0, horizon = 200, num_rollouts = 1):

        self.s_dim = H.shape[1]
        self.o_dim = H.shape[0]
        self.horizon = horizon
        self.num_rollouts = num_rollouts

        self.m = m      # Mass of the pendulum
        self.M = M      # Mass of the cart
        self.l = l      # Length of the pendulum
        self.g = g      # Acceleration due to gravity, reframed, now up is down so goal is zeros
        self.d = d      # Damping factor
        self.dt = dt    # How fast things change

        self.H = H
        self.Q = Q
        self.R = R
        self.proc_noise = np.random.multivariate_normal(np.zeros(self.s_dim), self.Q, [self.num_rollouts, self.horizon]).T
        self.obs_noise = np.random.multivariate_normal(np.zeros(self.o_dim), self.R, [self.num_rollouts, self.horizon]).T

        self.done = False
        self.t = 0
        self.s0 = s0
        self.s = s0

    def step_dot(self, u):
        x, v, theta, omega = self.s
        u = u[0]

        ddv = ((-self.m ** 2 * self.l ** 2 * self.g * np.cos(theta) * np.sin(theta) +
               self.m * self.l ** 2 * (self.m * self.l * omega ** 2 * np.sin(theta) - self.d * v) +
               self.m * self.l ** 2 * u) / (self.m * self.l ** 2 * (self.M + self.m * (1 - np.cos(theta) ** 2))))
        ddomega = (((self.m + self.M) * self.m * self.g * self.l * np.sin(theta) -
                self.m * self.l * np.cos(theta) * (self.m * self.l * omega ** 2 * np.sin(theta) - self.d * v) -
                self.m * self.l * np.cos(theta) * u) / (self.m * self.l ** 2 * (self.M + self.m * (1 - np.cos(theta) ** 2))))

        return np.array([v, ddv, omega, ddomega])

    def step(self, u):
        # Injecting noise into simulation:
        next_s = self.s + self.dt * self.step_dot(u) + self.proc_noise[:, self.t]
        obs = self.H @ next_s + self.obs_noise[:, self.t]

        self.s = next_s
        self.t += 1

        return next_s, obs

    def reset(self):
        self.proc_noise = np.random.multivariate_normal(np.zeros(self.s_dim), self.Q, [self.num_rollouts, self.horizon]).T
        self.obs_noise = np.random.multivariate_normal(np.zeros(self.o_dim), self.R, [self.num_rollouts, self.horizon]).T

        self.s = self.s0
        obs = self.H @ self.s + self.obs_noise[:, 0]

        self.done = False
        self.t = 0
        return self.s.copy(), obs

class DiffDrive:
    def __init__(self, dt, H, Q, R, s0, s_ref=None, u_ref=None, horizon=200, num_rollouts=1):
        self.s_dim = 3
        self.o_dim = H.shape[0]
        self.horizon = horizon
        self.num_rollouts = num_rollouts

        self.dt = dt
        self.H = H
        self.Q = Q
        self.R = R

        self.proc_noise = np.random.multivariate_normal(np.zeros(self.s_dim), self.Q, [self.num_rollouts, self.horizon]).T
        self.obs_noise = np.random.multivariate_normal(np.zeros(self.o_dim), self.R, [self.num_rollouts, self.horizon]).T

        self.done = False
        self.t = 0
        self.s0 = s0
        self.s = s0.copy()
        self.s_ref = s_ref
        self.u_ref = u_ref

    def step_dot(self, u):
        theta = self.s[2]
        v, omega = u[0], u[1]
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        dtheta = omega
        return np.array([dx, dy, dtheta])

    def step(self, u):
        next_s = self.s + self.dt * self.step_dot(u) + self.proc_noise[:, self.t]
        # NOTE: theta is left UNWRAPPED here on purpose, to stay consistent
        # with the unwrapped s_ref used by the controller. We only wrap
        # for plotting/observation, never inside the state used for control.
        obs = self.H @ next_s + self.obs_noise[:, self.t]

        self.s = next_s
        self.t += 1
        self.done = self.t >= self.horizon
        return next_s, obs

    def look_ahead(self, u):
        next_s = self.s + self.dt * self.step_dot(u) + self.proc_noise[:, self.t]
        obs = self.H @ next_s + self.obs_noise[:, self.t]
        return next_s, obs

    def reset(self):
        self.proc_noise = np.random.multivariate_normal(np.zeros(self.s_dim), self.Q, [self.num_rollouts, self.horizon]).T
        self.obs_noise = np.random.multivariate_normal(np.zeros(self.o_dim), self.R, [self.num_rollouts, self.horizon]).T
        self.s = self.s0.copy()
        obs = self.H @ self.s + self.obs_noise[:, 0]
        self.done = False
        self.t = 0
        return self.s.copy(), obs

class LinearSystem:
    def __init__(self, A, B, H, Q, R, s0, horizon = 200, num_rollouts = 1):

        self.s_dim = H.shape[1]
        self.o_dim = H.shape[0]
        self.horizon = horizon
        self.num_rollouts = num_rollouts

        self.A = A
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.proc_noise = np.random.multivariate_normal(np.zeros(self.s_dim), self.Q, [self.num_rollouts, self.horizon]).T
        self.obs_noise = np.random.multivariate_normal(np.zeros(self.o_dim), self.R, [self.num_rollouts, self.horizon]).T

        self.done = False
        self.t = 0
        self.s0 = s0
        self.s = s0

    def step(self, u):  # u has shape [dim, num_rollouts]
        next_s = self.A @ self.s + self.B @ u + self.proc_noise[:, self.t]
        obs = self.H @ next_s + self.obs_noise[:, self.t]

        self.s = next_s
        self.t += 1

        return next_s, obs

    def reset(self):
        self.proc_noise = np.random.multivariate_normal(np.zeros(self.s_dim), self.Q, [self.num_rollouts, self.horizon]).T
        self.obs_noise = np.random.multivariate_normal(np.zeros(self.o_dim), self.R, [self.num_rollouts, self.horizon]).T

        self.s = self.s0
        obs = self.H @ self.s + self.obs_noise[:, 0]

        self.done = False
        self.t = 0
        return self.s.copy(), obs

def wrap_to_pi(angle):
    """Wrap angle(s) to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def simulate(policy, task_description, env, s_ref=None, u_ref=None, horizon=1000, num_rollouts=1000):

    A, B, H, Q, R = task_description["model"]
    Cs, Cu, Cb = task_description["cost_rate"]
    angles = task_description["angles"]

    dim_s, dim_o, dim_m, dim_u = A.shape[0], H.shape[0], A.shape[0], B.shape[1]

    if s_ref is None:
        s_ref = np.zeros([dim_s, horizon])

    if u_ref is None:
        u_ref = np.zeros([dim_u, horizon])

    s, o = env.reset()

    G, K, L = policy["G"], policy["K"], policy["L"]

    ss = np.zeros([horizon, dim_s, num_rollouts])
    os = np.zeros([horizon, dim_o, num_rollouts])
    ms = np.zeros([horizon, dim_m, num_rollouts])
    us = np.zeros([horizon, dim_u, num_rollouts])

    sc = np.zeros([horizon, num_rollouts])
    uc = np.zeros([horizon, num_rollouts])
    ic = np.zeros([horizon])

    u = np.stack([u_ref[:, 0]] * num_rollouts, axis=1) + np.random.randn(dim_u, num_rollouts)
    mu = s-np.stack([s_ref[:, 0]] * num_rollouts, axis=1)
    mu[angles] = wrap_to_pi(mu[angles])

    for t in range(horizon):

        next_s, next_o = env.step(u)
        next_mu = G @ mu + K @ (next_o - np.stack([H @ s_ref[:, t]] * num_rollouts, axis=1))
        next_mu[angles] = wrap_to_pi(next_mu[angles])
        next_u = np.stack([u_ref[:, t]] * num_rollouts, axis=1) + L @ next_mu

        ss[t] = s
        os[t] = o
        ms[t] = mu
        us[t] = u - np.stack([u_ref[:, t]] * num_rollouts, axis=1)

        # Computing costs
        sc[t] = np.einsum("ki,kl,li->i", ss[t], Cs, ss[t], optimize=True)
        uc[t] = np.einsum("ki,kl,li->i", us[t], Cu, us[t], optimize=True)

        """
        if dim_s > 1:
            logdet_X = np.log(np.linalg.det(np.cov(ss[t])))
            logdet_M = np.log(np.linalg.det(np.cov(ms[t])))
            logdet_S = np.log(np.linalg.det(np.cov(np.vstack((ss[t], ms[t])))))
        else:
            logdet_X = np.log(np.cov(ss[t]))
            logdet_M = np.log(np.cov(ms[t]))
            logdet_S = np.log(np.linalg.det(np.cov(np.vstack((ss[t], ms[t])))))

        ic[t] = 0.5 * (logdet_X + logdet_M - logdet_S) / np.log(2)
        """
        s, o, mu, u = next_s, next_o, next_mu, next_u

    taus = {"states":ss, "actions":us, "estimates":ms, "observations":os}
    costs = {"state":np.mean(sc, axis=1), "action":np.mean(uc, axis=1), "inference":None}
    return taus, costs