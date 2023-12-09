"""Module to create a multi path channel model.

Classes:
        SCMMulti: Class to build a multi path channel model.

"""
import numpy as np
import modules.SCM3GPP.scm_helper as scm_helper


class SCMMulti:
    """Class to build a multi path channel model.

    This class defines a multi path channel model.

    Public Methods:

    Instance Variables:

    """

    def __init__(self, path_sigma=2.0, n_path=3):
        """Initialize multi path channel model.

        First, initialise all variables belonging to the multi path channel model.
        """
        self.path_sigma = path_sigma
        self.n_path = n_path

    def generate_channel(
        self,
        n_batches,
        n_coherence,
        n_antennas,
        rng=np.random.default_rng()
    ):
        """Generate multi path model parameters.

        Returns:
            A tuple (h, t) consisting of channels h with
                h.shape = (n_batches, n_coherence, n_antennas)
            and the first rows t of the covariance matrices with
                t.shape = (n_batches, n_antennas)
        """

        h = np.zeros([n_batches, n_coherence, n_antennas], dtype=np.complex64)
        t = np.zeros([n_batches, n_antennas], dtype=np.complex64)

        for i in range(n_batches):
            gains = rng.random(self.n_path)
            gains = gains / np.sum(gains, axis=0)
            angles = (rng.random(self.n_path) - 0.5) * 180

            h[i, :, :], t[i, :] = scm_helper.scm_channel(angles, gains, n_coherence, n_antennas, self.path_sigma, rng=rng)

        return h, t

    def get_config(self):
        config = {
            'path_sigma': self.path_sigma,
            'n_path': self.n_path
        }
        return config
