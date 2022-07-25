import numpy as np


class ATypeModel(object):
    def __init__(
            self,
            C_m=1,
            E_Na=50,
            E_K=-77,
            E_A=-80,
            E_L=-22,
            E_m=-73,
            g_Na=120,
            g_K=20,
            g_A=47.7,
            g_L=0.3
    ):
        self.C_m = C_m
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
        self.E_A = E_A
        self.E_m = E_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.g_A = g_A

        self.m_0 = self.m_inf(E_m)
        self.h_0 = self.h_inf(E_m)
        self.n_0 = self.n_inf(E_m) #.54
        self.a_0 = self.a_inf(E_m)
        self.b_0 = self.b_inf(E_m)

        print(self.n_0, self.m_0, self.h_0, self.a_0, self.b_0)

    @staticmethod
    def alpha_m(V):
        V_e = -34.7

        if V == V_e:
            return 3.8

        return 3.8 * 0.1 * (V - V_e) / (1 - np.exp(-(V - V_e) / 10))

    @staticmethod
    def beta_m(V):
        return 3.8 * 4 * np.exp(- (V + 59.7) / 18)

    @staticmethod
    def alpha_h(V):
        return 3.8 * 0.07 * np.exp(- (V + 53) / 20)

    @staticmethod
    def beta_h(V):
        return 3.8 * 1 / (1 + np.exp(- (V + 23) / 10))

    @staticmethod
    def alpha_n(V):
        V_e = -50.7

        if V == V_e:
            return 3.8 / 2 * 0.1

        return 3.8 / 2 * 0.01 * (V - V_e) / (1 - np.exp(-(V - V_e) / 10))

    @staticmethod
    def beta_n(V):
        return 3.8 / 2 * 0.125 * np.exp(-(V + 60.7) / 80)

    def m_inf(self, V):
        dt = 0.01
        m = 0
        for _ in range(1000):
            m += dt * (self.alpha_m(V) * (1 - m) - self.beta_m(V) * m)
        return m

    def n_inf(self, V):
        dt = 0.1
        n = 0
        for _ in range(1000):
            n += dt * (self.alpha_n(V) * (1 - n) - self.beta_n(V) * n)
        return n

    def h_inf(self, V):
        dt = 0.1
        h = 0

        for _ in range(1000):
            h += dt * (self.alpha_h(V) * (1 - h) - self.beta_h(V) * h)

        return h

    @staticmethod
    def a_inf(V):
        nominator = 0.0761 * np.exp((V + 99.22) / 31.84)
        denominator = 1 + np.exp((V + 6.17) / 28.93)
        return (nominator / denominator) ** (1.0/3)

    @staticmethod
    def a_tau(V):
        denominator = 1 + np.exp((V + 60.96) / 20.12)
        return 0.3632 + 1.158 / denominator

    @staticmethod
    def b_inf(V):
        denominator = (1 + np.exp((V + 58.3) / 14.54))**4
        return 1 / denominator

    @staticmethod
    def b_tau(V):
        denominator = 1 + np.exp((V - 55) / 16.027)
        return 1.24 + 2.678 / denominator

    def response(self, I_e, start=-200, end=600, dt=0.025):
        ts = np.arange(start, end, dt)

        I = np.zeros(len(ts))
        V = np.ones(len(ts)) * self.E_m

        # Channel coefficients
        m = np.ones(len(ts)) * self.m_0
        h = np.ones(len(ts)) * self.h_0
        n = np.ones(len(ts)) * self.n_0

        # A type coefficients
        a = np.ones(len(ts)) * self.a_0
        b = np.ones(len(ts)) * self.b_0

        for step, t in enumerate(ts[:-1]):
            I[step] = I_e(t)

            I_L = self.g_L * (V[step] - self.E_L)
            I_Na = self.g_Na * m[step] ** 3 * h[step] * (V[step] - self.E_Na)
            I_K = self.g_K * n[step] ** 4 * (V[step] - self.E_K)
            I_A = self.g_A * a[step] ** 3 * b[step] * (V[step] - self.E_A)

            V[step + 1] = dt / self.C_m * (I[step] - I_L - I_Na - I_K - I_A) + V[step]

            m[step + 1] = dt * (self.alpha_m(V[step + 1]) * (1 - m[step]) - self.beta_m(V[step + 1]) * m[step]) + m[step]
            h[step + 1] = dt * (self.alpha_h(V[step + 1]) * (1 - h[step]) - self.beta_h(V[step + 1]) * h[step]) + h[step]
            n[step + 1] = dt * (self.alpha_n(V[step + 1]) * (1 - n[step]) - self.beta_n(V[step + 1]) * n[step]) + n[step]

            a[step + 1] = dt * (self.a_inf(V[step + 1]) - a[step]) / self.a_tau(V[step + 1]) + a[step]
            b[step + 1] = dt * (self.b_inf(V[step + 1]) - b[step]) / self.b_tau(V[step + 1]) + b[step]

        return ts, V, I, m, h, n, a, b
