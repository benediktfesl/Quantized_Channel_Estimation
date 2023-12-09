import numpy as np
from modules import utils as ut
from scipy.special import erfinv, erf
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import csv
from modules.SCM3GPP.SCMMulti import SCMMulti
from modules.utils import toeplitz


def get_func_erfinv(t1, t2, prob):
    def f_erfinv(s):
        return erf(t2 / (np.sqrt(2) * s)) - erf(t1 / (np.sqrt(2) * s)) - prob
    return f_erfinv


def get_func_ls(t, probs, len):
    def f_ls(s):
        res = np.zeros(len)
        for ir in range(len):
            res[ir] = erf(t[ir] / (np.sqrt(2)*s)) - probs[ir]
        return res
    return f_ls


def get_func_newt(t, probs):
    len = probs.shape[0]
    def f_newt(s):
        res = np.zeros(len)
        for ir in range(len):
            res[ir] = erf(t[ir] / (np.sqrt(2)*s)) - probs[ir]
        return res
    return f_newt


def get_func_jacob(t):
    len = probs.shape[0]
    def f_jac(s):
        jac = np.zeros(len)
        for ir in range(len):
            jac[ir] = -np.sqrt(2/np.pi)*t[ir]*np.exp(-t[ir]**2 / (2*s)) / s**2
        return jac
    return f_jac


if __name__ == "__main__":
    n_bits = 3
    quantizer_type = 'uniform'
    n_dim = 64
    n_data_list = [10, 100, 1_000, 10_000, 100_000, 1_000_000]
    mc_runs = 10
    n_path = 1
    path_sigma = 2.0

    #generate 3gpp data
    channel_scm = SCMMulti(path_sigma=path_sigma, n_path=n_path)
    rng = np.random.default_rng(np.random.randint(1e9))

    norm_fac = 0.0
    norm_fac_corr = 0.0
    mse_list_real = list()
    mse_list_both = list()
    mse_list_unquant_both = list()
    mse_list_quant = list()

    iter_list = list()
    data_ = n_data_list.copy()
    data_.insert(0, 'data')
    iter_list.append(data_)
    iter_list.append(['avg_iter'])

    mse_list = list()
    data_ = n_data_list.copy()
    data_.insert(0, 'data')
    mse_list.append(data_)
    mse_list.append(['sampcov_quant'])
    mse_list.append(['sampcov_unquant'])
    mse_list.append(['Cov_est'])
    for n_data in n_data_list:
        _, toep = channel_scm.generate_channel(mc_runs, 1, n_dim, rng)
        mse_both = 0.0
        mse_unquant_both = 0.0
        mse_quant = 0.0
        iter_mean = 0.0
        norm_fac = 0.0
        for i in range(mc_runs):
            print(f'samples={n_data} | runs={i}/{mc_runs}', end='\r')
            cov = toeplitz(toep[i, :]).T  # get full cov matrix
            cov += np.abs(rng.standard_normal()) * np.eye(n_dim)
            norm_fac += np.sum(np.abs(cov)**2) / mc_runs
            x = sqrtm(cov) @ (rng.standard_normal(size=(n_data,n_dim,1)) + 1j*rng.standard_normal(size=(n_data,n_dim,1))) / np.sqrt(2)
            sigma2_avg = np.real(np.trace(cov) / n_dim)
            snr_equivalent = 10*np.log10(1 / (sigma2_avg - 1))
            #Correlation matrix estimation:
            # get quantizer
            quantizer_1bit = ut.get_quantizer([snr_equivalent], n_bits=1, quantizer_type=quantizer_type)
            thresholds_1bit = quantizer_1bit[snr_equivalent][0]
            quant_labels_1bit = quantizer_1bit[snr_equivalent][1]
            r_1bit = ut.quant(x, n_bits=1, thresholds=thresholds_1bit, quant_labels=quant_labels_1bit)
            corr_est = np.zeros_like(cov)
            for isamp in range(n_data):
                corr_est += r_1bit[isamp] @ r_1bit[isamp].T.conj() / n_data
            corr_est = np.sin(np.pi/2 * corr_est.real) + 1j*np.sin(np.pi/2 * corr_est.imag)
            # get quantizer
            quantizer = ut.get_quantizer([snr_equivalent], n_bits, quantizer_type=quantizer_type)
            thresholds = quantizer[snr_equivalent][0]
            quant_labels = quantizer[snr_equivalent][1]
            r = np.squeeze(ut.quant(x, n_bits=n_bits, thresholds=thresholds, quant_labels=quant_labels))
            if n_data == 1:
                r = np.expand_dims(r, 0)

            sigma2_real = np.real(np.diag(cov)) / 2
            sigma2_both_hat = np.zeros([n_dim])
            for idim in range(n_dim):
                sigma2_est_all = list()
                # initial estimate
                thres = thresholds[(thresholds.shape[0] - 1) // 2 + 1:]
                probs = np.zeros([int(2 ** (n_bits - 1) - 1), 2])
                for b in range(2 ** (n_bits - 1) - 1):
                    probs[b, 0] = np.sum(np.abs(r[:, idim].real) < thres[b]) / n_data
                    probs[b, 1] = np.sum(np.abs(r[:, idim].imag) < thres[b]) / n_data
                probs = np.clip(probs, 1/n_data, (n_data-1)/n_data)
                sigma2_real_hat = (thres[-1] / (np.sqrt(2) * erfinv(probs[-1, 0]))) ** 2
                sigma2_imag_hat = (thres[-1] / (np.sqrt(2) * erfinv(probs[-1, 1]))) ** 2
                x0 = np.sqrt((sigma2_real_hat + sigma2_imag_hat) / 2)
                x0 = np.clip(x0, 0.5, 3.5)
                #Gauss newton to solve nonlinear LS
                thres = np.concatenate((thres, thres), axis=0)
                probs = np.concatenate((probs[:, 0], probs[:, 1]), axis=0)
                func = get_func_newt(thres, probs)
                func_jac = get_func_jacob(thres)
                sigma2_ls_real, it = ut.gauss_newt_solve(func, func_jac, x0)
                iter_mean += it / (n_dim*mc_runs)
                sigma2_ls_real = sigma2_ls_real ** 2
                sigma2_both_hat[idim] = 2*sigma2_ls_real
            cov_est_both = np.diag(np.sqrt(sigma2_both_hat)) @ corr_est @ np.diag(np.sqrt(sigma2_both_hat))
            mse_both += np.sum(np.abs(cov - cov_est_both)**2)

            # unquantized estimate
            samp_cov = np.zeros_like(cov)
            for isamp in range(n_data):
                samp_cov += x[isamp] @ x[isamp].T.conj() / n_data
            mse_unquant_both += np.sum(np.abs(samp_cov - cov)**2)

            # quantized covariance matrix
            Cr_samp = np.zeros_like(cov)
            for isamp in range(n_data):
                Cr_samp += np.expand_dims(r[isamp], 1) @ np.expand_dims(r[isamp], 1).T.conj() / n_data
            mse_quant += np.sum(np.abs(Cr_samp - cov) ** 2)

        mse_list_both.append(mse_both / mc_runs / norm_fac)
        mse_list_unquant_both.append(mse_unquant_both / mc_runs / norm_fac)
        mse_list_quant.append(mse_quant/ mc_runs / norm_fac)
        mse_list[-1].append(mse_both/ mc_runs / norm_fac)
        mse_list[-2].append(mse_unquant_both/ mc_runs / norm_fac)
        mse_list[-3].append(mse_quant/ mc_runs / norm_fac)
        iter_list[-1].append(iter_mean)
    plt.loglog(n_data_list, mse_list_both)
    plt.loglog(n_data_list, mse_list_unquant_both)
    plt.loglog(n_data_list, mse_list_quant)
    plt.legend(["Real+imag avg", "Unquant sampcov", "Quant sampcov"])
    plt.title(f"NMSE of estimated {n_dim}x{n_dim} cov ({n_bits}bit, {mc_runs}MC runs)")
    plt.xlabel('data samples')
    plt.ylabel('NMSE')
    plt.savefig(f'results/cov_est_quant/3gpp_path={n_path}_dim={n_dim}_bits={n_bits}_mcs={mc_runs}_{quantizer_type}.png')

    mse_list = [list(i) for i in zip(*mse_list)]
    iter_list = [list(i) for i in zip(*iter_list)]
    print(mse_list)
    file_name = f'./results/cov_est_quant/3gpp_path={n_path}_dim={n_dim}_bits={n_bits}_mcs={mc_runs}_{quantizer_type}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)
    file_name = f'./results/cov_est_quant/3gpp_path={n_path}_dim={n_dim}_bits={n_bits}_mcs={mc_runs}_{quantizer_type}_iter.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(iter_list)