import numpy as np
from modules.utils import crandn


def scm_channel(
    angles,
    weights,
    n_coherence,
    n_antennas,
    sigma=2.0,
    rng=np.random.default_rng()
):
    (h, t) = chan_from_spectrum(n_coherence, n_antennas, angles, weights, sigma, rng=rng)
    return h, t


def spectrum(u, angles, weights, sigma=2.0):
    u = (u + np.pi) % (2 * np.pi) - np.pi
    theta = np.degrees(np.arcsin(u / np.pi))
    v = _laplace(theta, angles, weights, sigma) \
        + _laplace(180 - theta, angles, weights, sigma)

    return np.degrees(2 * np.pi * v / np.sqrt(np.pi ** 2 - u ** 2))


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
    n_antennas,
    angles,
    weights,
    sigma=2.0,
    rng=np.random.default_rng()
):

    o_f = 100  # oversampling factor (ideally, would use continuous freq. spectrum...)
    n_freq_samples = o_f * n_antennas

    # Sample the spectrum which is defined in equation (78) with epsilon, try
    # to avoid sampling at -pi and pi, thus avoiding dividing by zero.
    epsilon = 1 / 3
    lattice = np.arange(epsilon, n_freq_samples+epsilon) / n_freq_samples * 2 * np.pi - np.pi #sampled between -pi,+pi
    fs = spectrum(lattice, angles, weights, sigma)
    fs = np.reshape(fs, [len(fs), 1])

    # Avoid instabilities due to almost infinite energy at some frequencies
    # (this should only happen at "endfire" of a uniform linear array where --
    # because of the arcsin-transform -- the angular psd grows to infinity).
    almost_inf_threshold = np.max([1, n_freq_samples])  # use n_freq_samples as threshold value...
    almost_inf_freqs = np.absolute(fs) > almost_inf_threshold
    # this should not/only rarely be entered due to the epsilon above; one might even want to increase the threshold
    # to, e.g., 30 * almost_inf_threshold

    # if any(np.absolute(fs) > 20 * almost_inf_threshold):
    #     print("almost inf: ", fs[almost_inf_freqs])

    fs[almost_inf_freqs] = almost_inf_threshold  # * np.exp(1j * np.angle(fs[almost_inf_freqs])) # only real values

    if np.sum(fs) > 0:
        fs = fs / np.sum(fs) * n_freq_samples  # normalize energy

    x = crandn(n_freq_samples, n_coherence, rng=rng)

    h = np.fft.ifft(np.sqrt(fs)*x, axis=0) * np.sqrt(n_freq_samples)
    h = h[0:n_antennas, :]

    # t is the first row of the covariance matrix of h (which is Toeplitz and Hermitian)
    t = np.fft.fft(fs, axis=0) / n_freq_samples
    t = t[0:n_antennas]
    t = np.reshape(t, n_antennas)

    return h.T, t
