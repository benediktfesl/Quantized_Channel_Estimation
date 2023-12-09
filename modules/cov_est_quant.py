import numpy as np
from modules import uniform_quantizer as quant
from modules import utils as ut
from scipy.special import erfinv, erf


def get_func_newt(t, probs):
    len = probs.shape[0]
    def f_newt(s):
        res = np.zeros(len)
        for ir in range(len):
            res[ir] = erf(t[ir] / (np.sqrt(2)*s)) - probs[ir]
        return res
    return f_newt


def get_func_jacob(t):
    len = t.shape[0]
    def f_jac(s):
        jac = np.zeros(len)
        for ir in range(len):
            jac[ir] = -np.sqrt(2/np.pi)*t[ir]*np.exp(-t[ir]**2 / (2*s)) / s**2
        return jac
    return f_jac


def cplx_1bit(inp):
    return 1 / np.sqrt(2) * (np.sign(np.real(inp)) + 1j * np.sign(np.imag(inp)))


def est_cov_from_quant(x, n_bits, thresholds, resp, nk, x0_vec=None):
    """
    Parameters
    ----------
    x: quantized data samples of the form (n_samples, n_dim)
    n_bits: number of quantization bits
    thresholds: thresholds of quantizer

    Returns
    -------
    cov: estimate of unquantized cov
    """
    n_data, n_dim = x.shape
    #cov_est = np.zeros([n_dim, n_dim], dtype=x.dtype)
    x_1bit = cplx_1bit(x)
    corr_est = np.dot(resp * x_1bit.T, x_1bit.conj()) / nk
    #corr_est.flat[::corr_est.shape[-1] + 1] += 1e-6
    #corr_est = np.diag(1 / np.sqrt(np.diag(corr_est))) @ corr_est @ np.diag(1 / np.sqrt(np.diag(corr_est)))
    #corr_est = np.zeros_like(cov_est)
    #for isamp in range(n_data):
    #    corr_est += x_1bit[isamp] @ x_1bit[isamp].T.conj() / n_data
    #eigvals, Q = np.linalg.eigh(corr_est)
    corr_est = np.sin(np.pi / 2 * np.real(corr_est)) + 1j * np.sin(np.pi / 2 * np.imag(corr_est))
    #eigvals, Q = np.linalg.eigh(corr_est)
    #eigvals[eigvals < 1e-6] = 1e-6
    #corr_est = Q @ np.diag(eigvals) @ Q.conj().T
    #resp[np.argmax(resp)] = 1.0
    #resp[resp < 1.0] = 0.0
    #n_data = x_max.shape[0]
    thres = thresholds[(thresholds.shape[0] - 1) // 2 + 1:]
    thres = np.concatenate((thres, thres), axis=0)
    sigma2_both_hat = np.zeros([n_dim])
    for idim in range(n_dim):
        # initial estimate
        probs = np.zeros([int(2 ** (n_bits - 1) - 1), 2])
        for b in range(2 ** (n_bits - 1) - 1):
            #test = resp * (np.abs(x[:, idim].real) < thres[b])
            probs[b, 0] = np.sum(resp * (np.abs(x[:, idim].real) < thres[b])) / nk
            probs[b, 1] = np.sum(resp * (np.abs(x[:, idim].imag) < thres[b])) / nk
        probs = np.clip(probs, 1 / nk, (nk - 1) / nk)
        #sigma2_real_hat = (thres[-1] / (np.sqrt(2) * erfinv(probs[-1, 0]))) ** 2
        #sigma2_imag_hat = (thres[-1] / (np.sqrt(2) * erfinv(probs[-1, 1]))) ** 2
        if x0_vec is None:
            #x0 = np.sqrt((sigma2_real_hat + sigma2_imag_hat) / 2)
            x0 = np.array(1.0)
        else:
            x0 = np.real(x0_vec[idim])
        # Gauss newton to solve nonlinear LS
        probs = np.concatenate((probs[:, 0], probs[:, 1]), axis=0)
        func = get_func_newt(thres, probs)
        func_jac = get_func_jacob(thres)
        sigma2_ls_real = ut.gauss_newt_solve(func, func_jac, x0)[0] ** 2
        if np.isnan(sigma2_ls_real):
            sigma2_ls_real = 1.0
        sigma2_both_hat[idim] = np.clip(2 * sigma2_ls_real, 0, np.inf)
    cov_est = np.diag(np.sqrt(sigma2_both_hat)) @ corr_est @ np.diag(np.sqrt(sigma2_both_hat))
    #np.fill_diagonal(cov_est, sigma2_both_hat)
    return cov_est