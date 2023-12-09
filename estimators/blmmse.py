import numpy as np
from modules.utils import toeplitz
from scipy import linalg as scilin
from modules import uniform_quantizer as quant_uni
from modules import lloyd_max_quantizer as quant_lloyd

def mp_eval(obj, y, toep, h_true, genie, A=None, n_bits=1, quantizer_type=None, quantizer=None, Cr=None):
    if genie:
        hest = obj.estimate_genie(y, toep, A, n_bits, quantizer_type, quantizer, Cr)
    else:
        hest = obj.estimate_global(y, toep, A, n_bits, quantizer_type, quantizer, Cr)
    return hest


class BLMMSE:
    def __init__(self, snr):
        self.snr = snr
        self.rho = 10 ** (0.1 * snr)
        self.sigma2 = 1 / self.rho

    def estimate_genie(self, y, t, A=None, n_bits=1, quantizer_type='uniform', quantizer=None, Cr=None):
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
                inner_real = np.real(Psi_12 @ np.real(Cy) @ Psi_12)
                inner_imag = np.real(Psi_12 @ np.imag(Cy) @ Psi_12)
                inner_real[inner_real > 1] = 1.0
                inner_imag[inner_imag > 1] = 1.0
                inner_real[inner_real < -1] = -1.0
                inner_imag[inner_imag < -1] = -1.0
                Cr = 2 / np.pi * (np.arcsin(inner_real) + 1j * np.arcsin(inner_imag))
                hest[b, :] = C @ A_eff.conj().T @ np.linalg.solve(Cr, y[b, :])
                #hest[b, :] = prod @ y[b, :]
        elif n_bits == 'inf' or n_bits == np.inf:
            for b in range(n_batches):
                C = toeplitz(t[b, :]).T  # get full cov matrix
                CAh = C @ A.conj().T
                Cy = A @ CAh + 1 / self.rho * np.eye(A.shape[0])
                Cinv = np.linalg.pinv(Cy)
                hest[b, :] = CAh @ Cinv @ y[b, :]
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
                Cr = A_buss[0, 0]**2 * Cy + (1 - A_buss[0, 0]**2) * np.diag(np.diag(Cy))
                hest[b, :]  = C @ A_eff.conj().T @ scilin.solve(Cr, y[b, :], assume_a='pos')
        return hest


    def estimate_global(self, y, C, A=None, n_bits=1, quantizer_type='uniform', quantizer=None, Cr=None):
        (n_batches, n_antennas) = y.shape
        if A is None:
            A = np.eye(n_antennas, dtype=complex)
        hest = np.zeros([y.shape[0], A.shape[1]], dtype=y.dtype)
        Cy = A @ C @ A.conj().T + 1 / self.rho * np.eye(A.shape[0])
        if n_bits == 1:
            Psi_12 = np.real(np.diag(1 / np.sqrt(np.diag(Cy))))
            A_eff = np.sqrt(2 / np.pi) * Psi_12 @ A
            inner_real = np.real(Psi_12 @ np.real(Cy) @ Psi_12)
            inner_imag = np.real(Psi_12 @ np.imag(Cy) @ Psi_12)
            inner_real[inner_real > 1] = 1.0
            inner_imag[inner_imag > 1] = 1.0
            inner_real[inner_real < -1] = -1.0
            inner_imag[inner_imag < -1] = -1.0
            Cr = 2 / np.pi * (np.arcsin(inner_real) + 1j * np.arcsin(inner_imag))
            prod = C @ A_eff.conj().T @ np.linalg.pinv(Cr)
        elif n_bits == 'inf' or n_bits == np.inf:
            Cinv = np.linalg.pinv(Cy)
            prod = C @ A.conj().T @ Cinv
        else:
            if quantizer_type == 'uniform':
                A_buss = quant_uni.get_Bussgang_matrix(snr_dB=self.snr, n_bits=n_bits, Cy=Cy)
            elif quantizer_type == 'lloyd':
                A_buss = quant_lloyd.get_Bussgang_matrix(n_bits=n_bits, Cy=Cy, quantizer=quantizer)
            else:
                A_buss = None
            A_eff = A_buss @ A
            if Cr is None:
                Cr = A_buss[0, 0]**2 * Cy + (1 - A_buss[0, 0]**2) * np.diag(np.diag(Cy))
            prod = C @ A_eff.conj().T @ np.linalg.pinv(Cr)
        for b in range(n_batches):
            hest[b, :] = prod @ y[b, :]
        return hest
