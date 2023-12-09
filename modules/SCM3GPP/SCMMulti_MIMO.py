"""Module to create a multi path channel model.

Classes:
        SCMMulti: Class to build a multi path channel model.

"""
import numpy as np
import SCM3GPP.scm_helper_MIMO as scm_helper


class SCMMulti:
    """Class to build a multi path channel model.

    This class defines a multi path channel model.

    Public Methods:

    Instance Variables:

    """

    def __init__(self, path_sigma_BS=2.0, path_sigma_MS=35.0, n_path=3):
        """Initialize multi path channel model.

        First, initialise all variables belonging to the multi path channel model.
        """
        self.path_sigma_BS = path_sigma_BS
        self.path_sigma_MS = path_sigma_MS
        self.n_path = n_path

    def generate_channel(
        self,
        n_batches,
        n_coherence,
        n_antennas_BS,
        n_antennas_MS,
        rng=np.random.default_rng()
    ):
        """Generate multi path model parameters.

        Function that generates the multi path model parameters for given inputs.
        """

        h = np.zeros([n_batches, n_coherence, n_antennas_BS*n_antennas_MS], dtype=np.complex64)
        t_BS = np.zeros([n_batches, n_antennas_BS], dtype=np.complex64)
        t_MS = np.zeros([n_batches, n_antennas_MS], dtype=np.complex64)

        #angles = np.linspace(-90,90,n_batches)
        for i in range(n_batches):
            gains = rng.random(self.n_path)
            gains = gains / np.sum(gains, axis=0)
            angles_BS = (rng.random(self.n_path) - 0.5) * 180
            angles_MS = (rng.random(self.n_path) - 0.5) * 180

            h[i, :, :], t_BS[i, :], t_MS[i,:] = scm_helper.chan_from_spectrum(n_coherence, n_antennas_BS, n_antennas_MS,
                                          angles_BS, angles_MS, gains, self.path_sigma_BS, self.path_sigma_MS, rng=rng)

        return h, t_BS, t_MS

    def get_config(self):
        config = {
            'path_sigma_BS': self.path_sigma_BS,
            'path_sigma_MS': self.path_sigma_MS,
            'n_path': self.n_path
        }
        return config
