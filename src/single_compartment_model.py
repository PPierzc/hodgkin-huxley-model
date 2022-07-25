import numpy as np


class SingleCompartmentModel(object):
    def __init__(
            self,
            C_m=1,
            E_Na=115,
            E_K=-12,
            E_L=10.6,
            E_m=0,
            g_Na=120,
            g_K=36,
            g_L=0.3
    ):
        self.C_m = C_m
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.E_m = E_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L

        self.m_0 = self.m_inf(0)
        self.h_0 = self.h_inf(0)
        self.n_0 = self.n_inf(0)

    @staticmethod
    def alpha_m(V):
        V_e = 25

        if V == V_e:
            return 1

        return 0.1 * (V - V_e) / (1 - np.exp(-(V - V_e) / 10))

    @staticmethod
    def beta_m(V):
        return 4 * np.exp(- V / 18)

    @staticmethod
    def alpha_h(V):
        return 0.07 * np.exp(- V / 20)

    @staticmethod
    def beta_h(V):
        return 1 / (1 + np.exp(- (V - 30) / 10))

    @staticmethod
    def alpha_n(V):
        V_e = 10

        if V == V_e:
            return 0.1

        return 0.01 * (V - V_e) / (1 - np.exp(-(V - V_e) / 10))

    @staticmethod
    def beta_n(V):
        return 0.125 * np.exp(-V / 80)

    def m_inf(self, V):
        dt = 0.025
        m = 0
        for _ in range(1000):
            m += dt * (self.alpha_m(V) * (1 - m) - self.beta_m(V) * m)
        return m

    def n_inf(self, V):
        dt = 0.025
        n = 0
        for _ in range(1000):
            n += dt * (self.alpha_n(V) * (1 - n) - self.beta_n(V) * n)
        return n

    def h_inf(self, V):
        dt = 0.025
        h = 0
        for _ in range(1000):
            h += dt * (self.alpha_h(V) * (1 - h) - self.beta_h(V) * h)
        return h

    def response(self, I_e, start=-200, end=600, dt=0.025):
        ts = np.arange(start, end, dt)

        I = np.zeros(len(ts))
        V = np.zeros(len(ts))

        m = np.ones(len(ts)) * self.m_0
        h = np.ones(len(ts)) * self.h_0
        n = np.ones(len(ts)) * self.n_0

        for step, t in enumerate(ts[:-1]):
            I[step] = I_e(t)

            I_L = self.g_L * (V[step] - self.E_L)
            I_Na = self.g_Na * m[step] ** 3 * h[step] * (V[step] - self.E_Na)
            I_K = self.g_K * n[step] ** 4 * (V[step] - self.E_K)

            V[step + 1] = dt / self.C_m * (I[step] - I_L - I_Na - I_K) + V[step]

            m[step + 1] = dt * (self.alpha_m(V[step]) * (1 - m[step]) - self.beta_m(V[step]) * m[step]) + m[step]
            h[step + 1] = dt * (self.alpha_h(V[step]) * (1 - h[step]) - self.beta_h(V[step]) * h[step]) + h[step]
            n[step + 1] = dt * (self.alpha_n(V[step]) * (1 - n[step]) - self.beta_n(V[step]) * n[step]) + n[step]

        return ts, V, I, m, h, n
