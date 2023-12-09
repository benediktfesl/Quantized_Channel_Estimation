import numpy as np
from utils import crandn
from scipy.linalg import toeplitz
from scipy.linalg import sqrtm


def scm_channel(
    angles_BS,
    angles_MS,
    weights,
    n_coherence,
    n_antennas_BS,
    n_antennas_MS,
    sigma_BS=2.0,
    sigma_MS=35.0,
    rng=np.random.default_rng()
):
    (h, t_BS, t_MS) = chan_from_spectrum(n_coherence, n_antennas_BS, n_antennas_MS, angles_BS, angles_MS, weights, sigma_BS, sigma_MS, rng=rng)
    return h, t_BS, t_MS


def spectrum(u, angles, weights, sigma=2.0):
    # spectrum according to eq (78)
    u = (u + np.pi) % (2 * np.pi) - np.pi
    theta = np.degrees(np.arcsin(u / np.pi))
    # now compute the power density function, which follows a laplace distribution
    v = _laplace(theta, angles, weights, sigma) \
        + _laplace(180 - theta, angles, weights, sigma)

    return np.degrees(2 * np.pi * v / np.sqrt(np.pi ** 2 - u ** 2))     # see eq (78)


def _laplace(theta, angles, weights, sigma=2.0):
    # The variance \sigma^2 of a Laplace density is \sigma^2 = 2 * scale_parameter^2.
    # Hence, the standard deviation \sigma is \sigma = sqrt(2) * scale_parameter.
    # The scale_parameter determines the Laplace density.
    # For an angular spread (AS) given in terms of a standard deviation \sigma
    # the scale parameter thus needs to be computed as scale_parameter = \sigma / sqrt(2)
    scale_parameter = sigma / np.sqrt(2)
    x_shifted = np.outer(theta, np.ones(angles.size)) - angles
    x_shifted = (x_shifted + 180) % 360 - 180
    v = weights / (2 * scale_parameter) * np.exp(-np.absolute(x_shifted) / scale_parameter)
    return v.sum(axis=1)


def chan_from_spectrum(
    n_coherence,
    n_antennas_BS,
    n_antennas_MS,
    angles_BS,
    angles_MS,
    weights,
    sigma_BS=2.0,
    sigma_MS=35.0,
    rng=np.random.default_rng()
):

    o_f = 100  # oversampling factor (ideally, would use continuous freq. spectrum...)
    n_freq_samples_BS = o_f * n_antennas_BS

    # Sample the spectrum which is defined in equation (78) with epsilon, try
    # to avoid sampling at -pi and pi, thus avoiding dividing by zero.
    epsilon = 1 / 3
    lattice = np.arange(epsilon, n_freq_samples_BS+epsilon) / n_freq_samples_BS * 2 * np.pi - np.pi #sampling between -pi,+pi
    fs_BS = spectrum(lattice, angles_BS, weights, sigma_BS)
    fs_BS = np.reshape(fs_BS, [len(fs_BS), 1])

    # Avoid instabilities due to almost infinite energy at some frequencies
    # (this should only happen at "endfire" of a uniform linear array where --
    # because of the arcsin-transform -- the angular psd grows to infinity).
    almost_inf_threshold = np.max([1, n_freq_samples_BS])  # use n_freq_samples as threshold value...
    almost_inf_freqs = np.absolute(fs_BS) > almost_inf_threshold
    # this should not/only rarely be entered due to the epsilon above; one might even want to increase the threshold
    # to, e.g., 30 * almost_inf_threshold

    # if any(np.absolute(fs) > 20 * almost_inf_threshold):
    #     print("almost inf: ", fs[almost_inf_freqs])

    fs_BS[almost_inf_freqs] = almost_inf_threshold  # * np.exp(1j * np.angle(fs[almost_inf_freqs])) # only real values

    if np.sum(fs_BS) > 0:
        fs_BS = fs_BS / np.sum(fs_BS) * n_freq_samples_BS  # normalize energy

    # do the same for MS:
    n_freq_samples_MS = o_f * n_antennas_MS
    # n_freq_samples_MS = n_freq_samples_BS
    lattice = np.arange(epsilon, n_freq_samples_MS+epsilon) / n_freq_samples_MS * 2 * np.pi - np.pi #sampling between -pi,+pi
    fs_MS = spectrum(lattice, angles_MS, weights, sigma_MS)
    fs_MS = np.reshape(fs_MS, [len(fs_MS), 1])
    almost_inf_threshold = np.max([1, n_freq_samples_MS])  # use n_freq_samples as threshold value...
    almost_inf_freqs = np.absolute(fs_MS) > almost_inf_threshold
    fs_MS[almost_inf_freqs] = almost_inf_threshold  # * np.exp(1j * np.angle(fs[almost_inf_freqs])) # only real values
    if np.sum(fs_MS) > 0:
        fs_MS = fs_MS / np.sum(fs_MS) * n_freq_samples_MS  # normalize energy

    # t is the first row of the covariance matrix of h (which is Toeplitz and Hermitian)
    t_BS = np.fft.fft(fs_BS, axis=0) / n_freq_samples_BS
    t_BS = t_BS[0:n_antennas_BS]
    t_BS = np.reshape(t_BS, n_antennas_BS)

    t_MS = np.fft.fft(fs_MS, axis=0) / n_freq_samples_MS
    t_MS = t_MS[0:n_antennas_MS]
    t_MS = np.reshape(t_MS, n_antennas_MS)

    C_BS = toeplitz(t_BS)
    C_MS = toeplitz(t_MS)
    C = np.kron(C_MS, C_BS)

    # safe version
    n_mult = n_antennas_BS * n_antennas_MS
    x = crandn(n_mult, n_coherence, rng=rng)
    h = np.zeros([n_mult, n_coherence], dtype=complex)
    for i in range(n_coherence):
        try:
            L = np.linalg.cholesky(C)
        except:
            L = sqrtm(C)
        h[:, i] = np.matmul(L, x[:, i])
    return h.T, t_BS, t_MS
