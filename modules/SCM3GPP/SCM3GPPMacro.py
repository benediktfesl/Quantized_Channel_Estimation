import numpy as np
import scm3gpp.scm_helper as scm_helper


class SCM3GPPMacro(object):

    def __init__(self):
        self.mu_AS = None
        self.eps_AS = None
        self.r_AS = None
        self.path_sigma = None
        # Delay spread
        self.mu_DS = None
        self.eps_DS = None
        self.r_DS = None
        self.eps_PL = None  # 10*path loss exponent
        self.n_paths = None

    def set_urban_macro15_deg(self):
        self.mu_AS = 1.18
        self.eps_AS = 0.21
        self.r_AS = 1.3
        self.path_sigma = 2.82843
        self.mu_DS = -6.18
        self.eps_DS = 0.18
        self.r_DS = 1.7
        self.eps_PL = 35
        self.n_paths = 6

    def generate_channel(
        self,
        n_batches,
        n_coherence,
        n_antennas,
        rng=np.random.random.__self__
    ):
        """Generate multi path model parameters.

        Function that generates the multi path model parameters for given inputs.
        """

        h = np.zeros([n_batches, n_coherence, n_antennas], dtype=np.complex64)
        t = np.zeros([n_batches, n_antennas], dtype=np.complex64)

        for i in range(n_batches):
            # user angle
            theta = (rng.rand() - 0.5) * 120

            # path delays
            DS = 10. ** (self.mu_DS + self.eps_DS * rng.randn())
            Tc = 1 / 3.84e6
            tau = -self.r_DS * DS * np.log(rng.rand(self.n_paths))
            tau = np.sort(tau) - tau.min()
            tau_quant = Tc / 16 * np.floor(tau / Tc * 16 + 0.5)

            # path powers
            exponent = -1. / DS * (self.r_DS - 1) / self.r_DS
            Z = rng.randn(self.n_paths) * 3  # per path shadow fading in dB
            p = np.exp(exponent * tau) * (10. ** (0.1 * Z))
            p = p / np.sum(p)

            # path AoDs
            AS = 10. ** (self.mu_AS + self.eps_AS * rng.randn())
            aodsm = rng.randn(self.n_paths) * self.r_AS * AS
            ixs = np.argsort(np.abs(aodsm))
            aodsm = aodsm[ixs]
            h[i, :, :], t[i, :] = scm_helper.scm_channel(theta + aodsm, p,
                                                         n_coherence, n_antennas, self.path_sigma)

            # pathloss
            minDist = 1000
            maxDist = 1500
            distance = rng.rand() * (maxDist - minDist) + minDist
            PL = self.eps_PL * np.log10(distance / maxDist)
            beta = 10 ** (-0.1 * PL)
            h[i, :, :] *= np.sqrt(beta)
            t[i, :] *= beta

        return h, t

    def get_config(self):
        config = {
            'mu_AS': self.mu_AS,
            'eps_AS': self.eps_AS,
            'r_AS': self.r_AS,
            'path_sigma': self.path_sigma,
            'mu_DS': self.mu_DS,
            'eps_DS': self.eps_DS,
            'r_DS': self.r_DS,
            'eps_PL': self.eps_PL,
            'n_paths': self.n_paths,
        }
        return config
