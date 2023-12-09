import numpy as np
from modules.utils import toeplitz
from modules import uniform_quantizer as quant_uni
from modules import lloyd_max_quantizer as quant_lloyd


def mp_eval(obj, y, toep, h_true, genie, A=None, n_bits=1, quantizer_type='uniform', quantizer=None):
    if genie:
        hest = obj.estimate_genie(y, toep, A, n_bits, quantizer_type, quantizer)
    else:
        hest = obj.estimate_global(y, toep, A, n_bits, quantizer_type, quantizer)
    return hest #return_mse(hest, h_true)


class LS:
    def __init__(self, snr):
        self.snr = snr
        self.rho = 10 ** (0.1 * snr)
        self.sigma2 = 1 / self.rho

    def estimate_genie(self, y, t, A=None, n_bits=1, quantizer_type='uniform', quantizer=None):
        (n_batches, n_antennas) = y.shape
        if A is None:
            A = np.eye(n_antennas, dtype=y.dtype)
        hest = np.zeros([y.shape[0], A.shape[1]], dtype=y.dtype)
        if n_bits == 1:
            for b in range(n_batches):
                C = toeplitz(t[b, :]).T  # get full cov matrix
                Cy = A @ C @ A.conj().T + 1 / self.rho * np.eye(A.shape[0])
                Psi_12 = np.real(np.diag(1 / np.sqrt(np.diag(Cy))))
                A_eff = np.sqrt(2 / np.pi) * Psi_12 @ A
                hest[b, :], _, _, _ = np.linalg.lstsq(A_eff, y[b, :], rcond=None)
        elif n_bits == 'inf' or n_bits == np.inf:
            for b in range(n_batches):
                hest[b, :] = np.linalg.lstsq(A, y[b, :])
        else:
            for b in range(n_batches):
                C = toeplitz(t[b, :]).T  # get full cov matrix
                Cy = A @ C @ A.T.conj() + 1 / self.rho * np.eye(A.shape[0])
                if quantizer_type == 'uniform':
                    A_buss = quant_uni.get_Bussgang_matrix(snr_dB=self.snr, n_bits=n_bits, Cy=Cy)
                elif quantizer_type == 'lloyd':
                    A_buss = quant_lloyd.get_Bussgang_matrix(n_bits=n_bits, Cy=Cy, quantizer=quantizer)
                else:
                    A_buss = None
                A_eff = A_buss @ A
                x = np.linalg.lstsq(A_eff, y[b, :], rcond=None)[0]
                if ~np.any(np.isnan(x)):
                    hest[b, :] = x
                else:
                    print('NaN')
                    hest[b, :] = np.zeros_like(hest[b,:])
        return hest

    def estimate_global(self, y, C, A=None, n_bits=1, quantizer_type='uniform', quantizer=None):
        (n_batches, n_antennas) = y.shape
        if A is None:
            A = np.eye(n_antennas, dtype=complex)
        Cy = A @ C @ A.conj().T + 1 / self.rho * np.eye(A.shape[0])
        if n_bits == 1:
            Psi_12 = np.real(np.diag(1 / np.sqrt(np.diag(Cy))))
            A_eff = np.sqrt(2 / np.pi) * Psi_12 @ A
        elif n_bits == 'inf' or n_bits == np.inf:
            A_eff = A
        else:
            if quantizer_type == 'uniform':
                A_buss = quant_uni.get_Bussgang_matrix(snr_dB=self.snr, n_bits=n_bits, Cy=Cy)
            elif quantizer_type == 'lloyd':
                A_buss = quant_lloyd.get_Bussgang_matrix(n_bits=n_bits, Cy=Cy, quantizer=quantizer)
            else:
                A_buss = None
            A_eff = A_buss @ A
        hest, _, _, _ = np.linalg.lstsq(A_eff, y.T, rcond=None)
        return hest.T
