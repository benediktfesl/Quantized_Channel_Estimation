import numpy as np
from scipy.stats import norm
from scipy import integrate


def get_rho_lloyd(snr_dB, n_bits):
    return n_bits*2**(-2*n_bits)


def get_Bussgang_matrix(n_bits, Cy, quantizer):
    tau = list(quantizer[0])
    labels = list(quantizer[1])
    tau.insert(0, -np.inf)
    tau.append(np.inf)
    Cy_diag_inv = 1 / np.diag(Cy)
    B = -labels[0] * np.exp(-tau[1]**2 * Cy_diag_inv)
    B += labels[int(2**n_bits-1)] * np.exp(-tau[int(2**n_bits-1)]**2 * Cy_diag_inv)
    for i in range(1, int(2**n_bits-1)):
        B += labels[i] * (np.exp(-tau[i]**2 * Cy_diag_inv) - np.exp(-tau[i+1]**2 * Cy_diag_inv))
    B /= np.sqrt(np.pi) * np.sqrt(np.diag(Cy))
    return np.diag(B)


def load_quantizer(snr, n_bits, sigmas_gmm=None, pk_gmm=None):
    sigma2 = 10 ** (-snr / 10)
    if sigmas_gmm is None:
        input_var = 0.5 * (1 + sigma2)
    else:
        input_var = 0.5 * (sigmas_gmm + sigma2)
        pk_gmm = np.real(pk_gmm)
    [thresholds, quant_labels, rho] = lloyd_max_quantizer(levels=int((2 ** n_bits) / 2), mean=0, variance=np.real(input_var), pk_gmm=pk_gmm)

    thresholds = thresholds[:-1]
    thresholds = np.concatenate((np.flip(-thresholds[1:]), thresholds), axis=0)
    quant_labels = np.concatenate((np.flip(-quant_labels), quant_labels), axis=0)

    return {snr: (thresholds, quant_labels, rho)}


def lloyd_max_quantizer(levels, mean, variance, max_iter=200, pk_gmm=None):
    """
    This routine clusters the positive portion of the Gaussian PDF of given mean
    and variance into required number of levels using an iterative update routine.
    The output is an array of converged cluster centroids.
    """
    max_int = np.clip(3 * np.max(variance), 0, 100)  # p-Value ~ 1 (< 1e-12)
    intervals = np.zeros([levels + 1])
    intervals[:-1] = np.linspace(0., max_int, levels)
    intervals[-1] = np.inf
    centroids = np.zeros(levels)
    #intervals[levels] = max_int
    thresh = 1e-5
    #diff = np.inf

    # while (update > thresh):
    for i in range(max_iter):
        intervals_prev = np.copy(intervals)
        for j in range(levels):
            try:
                if pk_gmm is None:
                    centroids[j] = integrate.quad(lambda x: x * norm.pdf(x, mean, variance ** 0.5), intervals[j], intervals[
                        j + 1])[0] / integrate.quad(lambda x: norm.pdf(x, mean, variance ** 0.5), intervals[j], intervals[j + 1])[0]
                else:
                    num = 0.0
                    denom = 0.0
                    for k in range(pk_gmm.shape[0]):
                        num += pk_gmm[k] * integrate.quad(lambda x: x * norm.pdf(x, mean, variance[k]**0.5), intervals[j], intervals[j + 1])[0]
                        denom += pk_gmm[k] * integrate.quad(lambda x: norm.pdf(x, mean, variance[k]**0.5), intervals[j], intervals[j + 1])[0]
                    centroids[j] = num / denom
            except ZeroDivisionError:
                centroids[j] = (intervals[j] + intervals[j+1]) / 2
                #break

        for j in range(levels - 1):
            intervals[j + 1] = (centroids[j + 1] + centroids[j]) / 2.

        if np.linalg.norm(intervals_prev[:-1] - intervals[:-1]) < thresh: #or diff_frac > 0.999:
            break

    #compute distortion factor:
    rho = 0.0
    if pk_gmm is None:
        for j in range(levels):
            rho += integrate.quad(lambda x: (x - centroids[j]) ** 2 * norm.pdf(x, mean, variance ** 0.5), intervals[j],intervals[j + 1])[0]
    else:
        for j in range(levels):
            for k in range(pk_gmm.shape[0]):
                rho += pk_gmm[k] * integrate.quad(lambda x: (x-centroids[j])**2 * norm.pdf(x, mean, variance[k] ** 0.5), intervals[j], intervals[j + 1])[0]

    return intervals, centroids, rho