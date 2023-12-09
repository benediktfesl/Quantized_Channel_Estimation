import numpy as np
import warnings
import torch
from scipy.stats import norm

def standard_quantization_step(n_bits):
    """
    Optimal quantization steps for uniform quantizer with standard Gaussian N(0,1) input.
    See J. Max, "Quantizing for minimum distortion," Table 2.
    """
    delta_uniform = dict([(1, 1.596), (2, 0.9957), (3, 0.5860), (4, 0.3352), (5, 0.1881), (6,0.1041),
                          (7,0.0569), (8,0.0308)])
    if n_bits <= 8:
        return delta_uniform[n_bits]
    else:
        """
        Use asymptotic approximation from 
        D. Hui and D. Neuhoff, “Asymptotic Analysis of Optimal Fixed-Rate Uniform Scalar Quantization,” Example 1.
        """
        warnings.warn('Optimal standard step size is unknown and thus approximated!')
        return 4 * np.sqrt(n_bits) * 2**(-n_bits)
        #return np.sqrt(n_bits) * 2**(-n_bits) * delta_uniform[1]
        #return np.sqrt(12*2**(-2*n_bits))


def standard_distortion_fac(n_bits):
    """
    Optimal quantization steps for uniform quantizer with standard Gaussian N(0,1) input.
    See J. Max, "Quantizing for minimum distortion," Table 2.
    """
    rho_uniform = dict([(1, 1 - 2 / np.pi), (2, 0.11885), (3, 0.037440), (4, 0.011535), (5, 0.0034914), (6,0.00104),
                          (7,0.00030433), (8,0.00008769)])
    if n_bits <= 8:
        return rho_uniform[n_bits]
    else:
        """
        Use asymptotic approximation from 
        D. Hui and D. Neuhoff, “Asymptotic Analysis of Optimal Fixed-Rate Uniform Scalar Quantization,” Example 1.
        """
        warnings.warn('Optimal standard distortion factor is unknown and thus approximated!')
        return get_rho_uniform(np.inf, n_bits)


def get_uniform_quant_step(snr_dB, n_bits):
    return np.sqrt((1 + 10 ** (-snr_dB / 10))/2) * standard_quantization_step(n_bits)


def get_uniform_quant_step_torch(snr_dB, n_bits):
    return torch.sqrt((1 + 10 ** (-snr_dB / 10))/2) * standard_quantization_step(n_bits)


def get_rho_uniform(snr_dB, n_bits):
    #return n_bits * 2 ** (-2 * n_bits)
    delt = get_uniform_quant_step(snr_dB, n_bits)
    rho = delt**2 / 12
    rho += np.exp(-2**(2 * n_bits - 3) * delt**2) / (2**(n_bits-1.5) * delt)**3 / np.sqrt(np.pi)
    return rho


def get_Bussgang_matrix(snr_dB, n_bits, Cy):
    if n_bits == np.inf:
        return np.eye(Cy.shape[-1])
    elif n_bits == 1:
        B = np.sqrt(2 / np.pi) * 1 / np.sqrt(np.diag(Cy))
    else:
        delta = get_uniform_quant_step(snr_dB, n_bits)
        Cy_diag_inv = 1 / np.diag(Cy)
        B = np.zeros([Cy.shape[0]], dtype=complex)
        for i in range(1, int(2**n_bits)):
            B += np.exp(-delta**2 * (i - 2**n_bits / 2)**2 * Cy_diag_inv)
        B *= delta / np.sqrt(np.pi) / np.sqrt(np.diag(Cy))
    return np.diag(B)


def get_Bussgang_matrix_torch(snr_dB, n_bits, Cy, device):
    if n_bits == float('inf'):
        return torch.eye(Cy.shape[-1], device=device, dtype=torch.complex128)
    elif n_bits == 1:
        B = torch.sqrt(2 / torch.tensor(np.pi)) * (1 / torch.sqrt(torch.diag(Cy)))
    else:
        delta = get_uniform_quant_step(snr_dB, n_bits)  # Assuming you have implemented this function
        Cy_diag_inv = 1 / torch.diag(Cy)
        B = torch.zeros([Cy.shape[0]], dtype=torch.complex128, device=device)
        for i in range(1, int(2**n_bits)):
            B += torch.exp(-delta**2 * (i - 2**n_bits / 2)**2 * Cy_diag_inv)
        B *= delta / torch.sqrt(torch.tensor(np.pi)) / torch.sqrt(torch.diag(Cy))
    return torch.diag(B)


def get_Bussgang_matrix_diag(snr_dB, n_bits, diagCy):
    B = torch.zeros_like(diagCy)
    for ibatch in range(diagCy.shape[0]):
        delta = get_uniform_quant_step_torch(snr_dB[ibatch], n_bits)
        Cy_diag_inv = 1 / diagCy[ibatch]
        for i in range(1, int(2**n_bits)):
            B[ibatch] +=  torch.exp(-delta**2 * (i - 2**n_bits / 2)**2 * Cy_diag_inv)
        B[ibatch] *= (delta / np.sqrt(torch.pi) / torch.sqrt(diagCy[ibatch]))
    return B


def get_Bussgang_matrix_diag_fast(snr_dB, n_bits, diagCy):
    batch_size = diagCy.shape[0]
    delta = get_uniform_quant_step_torch(snr_dB, n_bits)
    Cy_diag_inv = 1 / diagCy
    i_values = torch.arange(1, int(2**n_bits), device=diagCy.device, dtype=torch.float)
    B = torch.zeros(batch_size, device=diagCy.device)
    for i in i_values:
        exponent = -delta**2 * (i - 2**n_bits / 2)**2 * Cy_diag_inv
        B += torch.exp(exponent)
    B *= (delta / torch.sqrt(torch.tensor(torch.pi, dtype=torch.float, device=diagCy.device)) / torch.sqrt(diagCy))
    return B


def get_quantized_variance(sigma2, quantizer):
    sigma2 = sigma2 / 2 #account for complex distribution
    thresh = quantizer[0]
    labels = quantizer[1]
    if len(sigma2.shape) < 2:
        sigma2 = np.expand_dims(sigma2, 0)
    batches, dims = sigma2.shape
    res = np.zeros_like(sigma2)
    for b in range(batches):
        for d in range(dims):
            res[b, d] += labels[0]**2 * norm.cdf(thresh[0] / np.sqrt(sigma2[b, d]))
            res[b, d] += labels[-1]**2 * (1 - norm.cdf(thresh[-1] / np.sqrt(sigma2[b, d])))
            for i in range(1, labels.shape[0]-1):
                res[b, d] += labels[i]**2 *(norm.cdf(thresh[i] / np.sqrt(sigma2[b, d])) - norm.cdf(thresh[i-1] / np.sqrt(sigma2[b, d])))
    return 2 * np.squeeze(res)


def get_quantized_variance_torch(sigma2, quantizer, device=None):
    sigma2 = sigma2 / 2 #account for complex distribution
    thresh = quantizer[0]
    labels = quantizer[1]
    if len(sigma2.shape) < 2:
        sigma2 = torch.unsqueeze(sigma2, 0)
    batches, dims = sigma2.shape
    sigma2 = sigma2.cpu().detach().numpy()
    res = np.zeros_like(sigma2)
    for b in range(batches):
        for d in range(dims):
            res[b, d] += labels[0]**2 * norm.cdf(thresh[0] / np.sqrt(sigma2[b, d]))
            res[b, d] += labels[-1]**2 * (1 - norm.cdf(thresh[-1] / np.sqrt(sigma2[b, d])))
            for i in range(1, labels.shape[0]-1):
                res[b, d] += labels[i]**2 *(norm.cdf(thresh[i] / np.sqrt(sigma2[b, d])) - norm.cdf(thresh[i-1] / np.sqrt(sigma2[b, d])))
    return torch.as_tensor(2 * np.squeeze(res), device=device, dtype=torch.complex128)


def get_Cr(Cy, n_bits, snr=None, quantizer=None):
    if len(Cy.shape) < 3:
        Cy = np.expand_dims(Cy, axis=0)
    Cr = np.zeros_like(Cy)
    n_comp = Cy.shape[0]
    if n_bits == 1:
        for i in range(n_comp):
            Psi_12 = np.real(np.diag(1 / np.sqrt(np.diag(Cy[i, :, :]))))
            inner_real = np.real(Psi_12 @ np.real(Cy[i, :, :]) @ Psi_12)
            inner_imag = np.real(Psi_12 @ np.imag(Cy[i, :, :]) @ Psi_12)
            inner_real[inner_real > 1] = 1.0
            inner_imag[inner_imag > 1] = 1.0
            inner_real[inner_real < -1] = -1.0
            inner_imag[inner_imag < -1] = -1.0
            Cr[i, :, :] = 2 / np.pi * (np.arcsin(inner_real) + 1j * np.arcsin(inner_imag))
        return np.squeeze(Cr)
    elif n_bits == np.inf:
        return np.squeeze(Cy)
    else:
        for i in range(n_comp):
            A_buss = get_Bussgang_matrix(snr, n_bits, Cy[i])
            diagCr = get_quantized_variance(np.diag(Cy[i]), quantizer)
            Cr[i] = np.mean(np.diag(A_buss))**2 * Cy
            np.fill_diagonal(Cr[i], diagCr)
        return np.squeeze(Cr)


def get_Cr_torch(Cy, n_bits, snr=None, quantizer=None, device=None):
    if len(Cy.shape) < 3:
        Cy = torch.unsqueeze(Cy, dim=0)
    Cr = torch.zeros_like(Cy)
    n_comp = Cy.shape[0]
    if n_bits == 1:
        for i in range(n_comp):
            Psi_12 = torch.real(torch.diag(1 / torch.sqrt(torch.diag(Cy[i, :, :]))))
            inner_real = torch.real(Psi_12 @ torch.real(Cy[i, :, :]) @ Psi_12)
            inner_imag = torch.real(Psi_12 @ torch.imag(Cy[i, :, :]) @ Psi_12)
            inner_real[inner_real > 1] = 1.0
            inner_imag[inner_imag > 1] = 1.0
            inner_real[inner_real < -1] = -1.0
            inner_imag[inner_imag < -1] = -1.0
            Cr[i, :, :] = 2 / np.pi * (torch.asin(inner_real) + 1j * torch.asin(inner_imag))
        return np.squeeze(Cr)
    elif n_bits == np.inf:
        return torch.squeeze(Cy)
    else:
        for i in range(n_comp):
            A_buss = get_Bussgang_matrix_torch(snr, n_bits, Cy[i], device=device)
            diagCr = get_quantized_variance_torch(torch.diag(Cy[i]), quantizer, device)
            Cr[i] = torch.mean(torch.diag(A_buss))**2 * Cy
            Cr[i, range(len(Cr[i])), range(len(Cr[i]))] = diagCr
        return torch.squeeze(Cr)