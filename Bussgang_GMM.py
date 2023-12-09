import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import multiprocessing as mp
from modules.SCM3GPP.SCMMulti import SCMMulti
from estimators.blmmse import BLMMSE, mp_eval
from estimators.LS import LS
from modules.gmm_cplx_bussgang import Gmm_nbit
import datetime
import csv
import modules.utils as ut
from copy import deepcopy
import joblib
from modules.uniform_quantizer import get_Bussgang_matrix, get_Cr

def mp_gmm(obj, *args):
    return obj.estimate_from_y(*args)

def mp_gmm_LS(obj, *args):
    return obj.estimate_from_y_LS(*args)

def mp_LS_genie(obj, *args):
    return obj.estimate_genie(*args)

def mp_LS_global(obj, *args):
    return obj.estimate_global(*args)

if __name__ == "__main__":
    n_processes = int(mp.cpu_count() / 2)  # int(mp.cpu_count() / 2 - 1)
    print('Uses ' + str(n_processes) + ' processes')
    # prepare multiprocessing
    pool = mp.Pool(processes=n_processes)

    n_antennas = 64 # BS antennas
    n_components = 64 # GMM components
    n_summands_or_proba = 'all' # Number of GMM LMMSE that should be evaluated
    n_path = 1 # Number of propagation clusters of the 3GPP channel model
    n_pilots = 1 # Number of pilots
    n_bits = 2 # Number of quantization bits
    cov_type = 'full' # covariance type of the GMM {'full', 'toeplitz', 'circulant'}
    pilot_type = 'angle_amp' # Pilot type {'angle', 'angle_amp', 'rand', 'ones'}
    quantizer_type = 'uniform' # Quantizer type {'uniform', 'lloyd'}
    snrs = [-10, -5, 0, 5, 10, 15, 20] # SNR range to be evaluated

    params = dict()
    params['n_antennas'] = n_antennas
    params['n_comp'] = n_components
    params['n_bits'] = n_bits
    params['n_path'] = n_path
    params['cov_type'] = cov_type
    params['quantizer_type'] = quantizer_type
    params['zero_mean_gmm'] = True

    # choose which algorithms to evaluate
    eval_blmmse_genie = True # genie-Bussgang
    eval_blmmse_glob = True # global-Bussgang
    eval_blmmse_gmm = True # GMM-Bussgang
    eval_LS_glob = True # Bussgang least squares
    eval_rate = True # True if the rate lower bound should be evaluated in addition to the MSE

    n_channels = 110_000
    n_train_ch = 100_000 # training data
    n_val_ch = 10_000 # validation data

    mse_list = list()
    snrs_ = snrs.copy()
    snrs_.insert(0, 'SNR')
    mse_list.append(snrs_)

    rate_list = list()
    snrs_ = snrs.copy()
    snrs_.insert(0, 'SNR')
    rate_list.append(snrs_)

    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    # Create channel data by the 3GPP channel model
    params['model_type'] = '3gpp'
    params['n_path'] = n_path
    path_sigma = 2.0
    file_name_3gpp = 'results/saves/saved_data_ant=' + str(n_antennas) + '_model=' + str(params['model_type']) + \
                     '_paths=' + str(params['n_path']) + '_ntrain=' + str(n_train_ch) + '_nchan=' + \
                     str(n_channels) + '.npy'
    # try to load stored dataset, else create one and save it
    try:
        data = np.load(file_name_3gpp)
        channels = data[0] # channel data
        toep = data[1] # vectors to create the genie-covariances
    except FileNotFoundError:
        channel_scm = SCMMulti(path_sigma=path_sigma, n_path=n_path)
        rng = np.random.default_rng(np.random.randint(1e8))
        channels, toep = channel_scm.generate_channel(n_channels, 1, n_antennas, rng)
        channels = np.squeeze(channels)
        np.save(file_name_3gpp, (channels, toep))

    channel_scm = SCMMulti(path_sigma=path_sigma, n_path=n_path)
    rng = np.random.default_rng(np.random.randint(1e9))
    channels, toep = channel_scm.generate_channel(n_channels, 1, n_antennas, rng)
    channels = np.squeeze(channels)
    if len(channels.shape) == 1:
        channels = np.expand_dims(channels, 1)

    toep_train = toep[:n_train_ch]
    toep_val = toep[n_train_ch:n_train_ch+n_val_ch]
    channels_train = channels[:n_train_ch]
    channels_val = channels[n_train_ch:n_train_ch+n_val_ch]

    params['n_pilots'] = n_pilots
    params['n_train'] = n_train_ch
    params['n_val'] = n_val_ch

    # get pilot matrix
    A = ut.get_pilot_matrix(n_antennas, n_pilots, n_bits, pilot_type=pilot_type)

    # get quantizer
    quantizer = ut.get_quantizer(snrs, n_bits, quantizer_type=quantizer_type)

    # Compute sample covariance matrix with the training data
    cov = np.zeros([n_antennas, n_antennas], dtype=complex)
    for i in range(n_train_ch):
        cov = cov + np.expand_dims(channels_train[i, :], 1) @ np.expand_dims(channels_train[i, :].conj(), 0)
    cov = cov / n_train_ch

    # global-Bussgang
    if eval_blmmse_glob:
        mse_list.append(['blmmse_glob'])
        rate_list.append(['blmmse_glob_rstat'])
        eval_list_glob = list()
        # compute blmmse with global cov
        for snr in snrs:
            r = ut.get_observation_nbit(channels_val, snr, A, n_bits, quantizer[snr][0], quantizer[snr][1])
            if len(r.shape) == 1:
                r = np.expand_dims(r, 1)
            r_train = ut.get_observation_nbit(channels_train, snr, A, n_bits, quantizer[snr][0], quantizer[snr][1])
            if len(r_train.shape) == 1:
                r_train = np.expand_dims(r_train, 1)
            eval_list_glob.append([BLMMSE(snr), r, cov, channels_val, False, A, n_bits, quantizer_type, quantizer[snr]])
        # compute channel estimates in parallel
        res_glob_blmmse = pool.starmap(mp_eval, eval_list_glob)
        # evaluate MSE and achievable rate lower bound
        for it, res in enumerate(res_glob_blmmse):
            mse_act = np.sum(np.abs(res - channels_val) ** 2) / channels_val.size
            mse_list[-1].append(mse_act)
            if eval_rate:
                snr = snrs[it]
                Cy_act = cov + 10**(-snr/10) * np.eye(n_antennas, dtype=complex)
                Buss_glob = get_Bussgang_matrix(snr, n_bits, Cy_act)
                Cr = get_Cr(Cy_act, n_bits, snr, quantizer[snr])
                Cq_glob = Cr - Buss_glob @ cov @ Buss_glob.conj().T
                # evaluate statistical rate lower bound
                norm_fac = np.sum(np.abs(res)**2, axis=1)
                norm_fac_test = np.sum(np.abs(channels_val)**2, axis=1)
                for i in range(res.shape[0]):
                    res[i] /= norm_fac[i]
                inner = np.squeeze(np.expand_dims(res.conj(), 1) @ Buss_glob @ np.expand_dims(channels_val, 2))
                num = np.abs(np.mean(inner, axis=0)) ** 2
                den1 = np.var(inner, axis=0)
                den2 = np.real(np.squeeze(np.expand_dims(res.conj(), 1) @ Cq_glob @ np.expand_dims(res, 2)))
                den2 = np.mean(den2, axis=0)
                rate_glob2 = np.log2(1 + num / (den1 + den2))
                rate_list[-1].append(rate_glob2)

    # Bussgang least squares
    if eval_LS_glob:
        mse_list.append(['LS_glob'])
        rate_list.append(['LS_glob_stat'])
        eval_list_glob = list()
        # compute blmmse with global cov
        for snr in snrs:
            r = ut.get_observation_nbit(channels_val, snr, A, n_bits, quantizer[snr][0], quantizer[snr][1])
            eval_list_glob.append([LS(snr), r, cov, A, n_bits, quantizer_type, quantizer[snr]])
        # compute channel estimates in parallel
        res_glob_blmmse = pool.starmap(mp_LS_global, eval_list_glob)
        for it, res in enumerate(res_glob_blmmse):
            mse_act = np.sum(np.abs(res - channels_val) ** 2) / channels_val.size
            mse_list[-1].append(mse_act)
            if eval_rate:
                # compute rate lower bound: global
                snr = snrs[it]
                Cy_act = cov + 10**(-snr/10) * np.eye(n_antennas, dtype=complex)
                Buss_glob = get_Bussgang_matrix(snr, n_bits, Cy_act)
                Cr = get_Cr(Cy_act, n_bits, snr, quantizer[snr])
                Cq_glob = Cr - Buss_glob @ cov @ Buss_glob.conj().T
                Cq_inv = np.linalg.pinv(Cq_glob)
                rate_glob = 0.0
                rate_genie = 0.0
                for n_datai in range(res.shape[0]):
                    g_mf_h = res[n_datai].conj().T @ Buss_glob.conj().T @ Cq_inv
                    err = channels_val[n_datai] - res[n_datai]
                    rate_glob += np.real(
                        np.log2(1 + (np.abs(g_mf_h @ Buss_glob @ res[n_datai]) ** 2) / (
                                g_mf_h @ Cq_glob @ g_mf_h.conj().T + np.abs(g_mf_h @ Buss_glob @ err) ** 2)))
                rate_list[-2].append(rate_glob / res.shape[0])

                #evaluate statistical lower bound
                norm_fac = np.sum(np.abs(res)**2, axis=1)
                norm_fac_test = np.sum(np.abs(channels_val)**2, axis=1)
                for i in range(res.shape[0]):
                    res[i] /= norm_fac[i]
                inner = np.squeeze(np.expand_dims(res.conj(), 1) @ Buss_glob @ np.expand_dims(channels_val, 2))
                num = np.abs(np.mean(inner, axis=0)) ** 2
                den1 = np.var(inner, axis=0)
                den2 = np.real(np.squeeze(np.expand_dims(res.conj(), 1) @ Cq_glob @ np.expand_dims(res, 2)))
                den2 = np.mean(den2, axis=0)
                rate_glob2 = np.log2(1 + num / (den1 + den2))
                rate_list[-1].append(rate_glob2)

    # genie-Bussgang
    if eval_blmmse_genie:
        mse_list.append(['blmmse_genie'])
        rate_list.append(['blmmse_genie_rstat'])
        eval_list_genie = list()
        for snr in snrs:
            r = ut.get_observation_nbit(channels_val, snr, A, n_bits, quantizer[snr][0], quantizer[snr][1])
            eval_list_genie.append([BLMMSE(snr), r, toep_val, channels_val, True, A, n_bits, quantizer_type, quantizer[snr], None])
        # compute channel estimates in parallel
        res_genie_blmmse = pool.starmap(mp_eval, eval_list_genie)
        for it, res in enumerate(res_genie_blmmse):
            mse_act = np.sum(np.abs(res - channels_val) ** 2) / channels_val.size
            mse_list[-1].append(mse_act)
            if eval_rate:
                snr = snrs[it]
                Cy_act = cov + 10**(-snr/10) * np.eye(n_antennas, dtype=complex)
                Buss_glob = get_Bussgang_matrix(snr, n_bits, Cy_act)
                Cr = get_Cr(Cy_act, n_bits, snr, quantizer[snr])
                Cq_glob = Cr - Buss_glob @ cov @ Buss_glob.conj().T
                Cq_inv = np.linalg.pinv(Cq_glob)
                #evaluate statistical lower bound
                norm_fac = np.sum(np.abs(res)**2, axis=1)
                norm_fac_test = np.sum(np.abs(channels_val)**2, axis=1)
                for i in range(res.shape[0]):
                    res[i] /= norm_fac[i]
                inner = np.squeeze(np.expand_dims(res.conj(), 1) @ Buss_glob @ np.expand_dims(channels_val, 2))
                num = np.abs(np.mean(inner, axis=0)) ** 2
                den1 = np.var(inner, axis=0)
                den2 = np.real(np.squeeze(np.expand_dims(res.conj(), 1) @ Cq_glob @ np.expand_dims(res, 2)))
                den2 = np.mean(den2, axis=0)
                rate_glob = np.log2(1 + num / (den1 + den2))
                rate_list[-1].append(rate_glob)

    # evaluate rate with perfect CSI
    if eval_rate:
        rate_list.append(['perfect_rstat'])
        for snr in snrs:
            Cy_act = cov + 10 ** (-snr / 10) * np.eye(n_antennas, dtype=complex)
            Buss_glob = get_Bussgang_matrix(snr, n_bits, Cy_act)
            Cr = get_Cr(Cy_act, n_bits, snr, quantizer[snr])
            Cq_glob = Cr - Buss_glob @ cov @ Buss_glob.conj().T
            # evaluate statistical lower bound
            res = channels_val.copy()
            norm_fac = np.sum(np.abs(res) ** 2, axis=1)
            for i in range(res.shape[0]):
                res[i] /= norm_fac[i]
            inner = np.squeeze(np.expand_dims(res.conj(), 1) @ Buss_glob @ np.expand_dims(channels_val, 2))
            num = np.abs(np.mean(inner, axis=0)) ** 2
            den1 = np.var(inner, axis=0)
            den2 = np.real(np.squeeze(np.expand_dims(res.conj(), 1) @ Cq_glob @ np.expand_dims(res, 2)))
            den2 = np.mean(den2, axis=0)
            rate_glob = np.log2(1 + num / (den1 + den2))
            rate_list[-1].append(rate_glob)


    # GMM-Bussgang
    if eval_blmmse_gmm:
        file_name_gmm = f'results/saves/trained_gmm_ant=' + str(n_antennas) + '_comp=' + str(n_components) + \
                '_model=' + str(params['model_type']) + '_paths=' + str(params['n_path']) + \
                '_ntrain=' + str(n_train_ch) + f'_covtype={cov_type}_cplx_zeromean={params["zero_mean_gmm"]}.sav'
        try:
            gmm_est = joblib.load(file_name_gmm)
            print('Loading trained gmm successful.')
        except FileNotFoundError:
            gmm_est = Gmm_nbit(n_components=n_components, covariance_type=cov_type)
            print('Fit gmm...')
            gmm_est.fit(h=channels_train, zero_mean=params['zero_mean_gmm'])
            print('done.')
            joblib.dump(gmm_est, file_name_gmm)
        params['n_summands_or_proba'] = n_summands_or_proba
        rate_list.append(['gmm_rstat'])
        mse_list.append(['blmmse_gmm'])
        gmm_copy = deepcopy(gmm_est)
        gmm_list = list()
        for snr in snrs:
            r_val = ut.get_observation_nbit(channels_val, snr, A, n_bits, quantizer[snr][0], quantizer[snr][1])
            gmm_list.append([gmm_copy, r_val, snr, n_antennas, A, n_summands_or_proba, n_bits, quantizer_type, quantizer[snr]])
        res_gmm_blmmse = pool.starmap(mp_gmm, gmm_list)
        for it, res in enumerate(res_gmm_blmmse):
            mse_act = np.sum(np.abs(res - channels_val)**2) / channels_val.size
            mse_list[-1].append(mse_act)
            if eval_rate:
                #compute rate lower bound: global
                snr = snrs[it]
                Cy_act = cov + 10**(-snr/10) * np.eye(n_antennas, dtype=complex)
                Buss_glob = get_Bussgang_matrix(snr, n_bits, Cy_act)
                Cr = get_Cr(Cy_act, n_bits, snr, quantizer[snr])
                Cq_glob = Cr - Buss_glob @ cov @ Buss_glob.conj().T
                # evaluate statistical lower bound
                norm_fac = np.clip(np.sum(np.abs(res) ** 2, axis=1), 1e-1, np.inf)
                norm_fac_test = np.sum(np.abs(channels_val) ** 2, axis=1)
                for i in range(res.shape[0]):
                    res[i] /= norm_fac[i]
                inner = np.squeeze(np.expand_dims(res.conj(), 1) @ Buss_glob @ np.expand_dims(channels_val, 2))
                num = np.abs(np.mean(inner, axis=0)) ** 2
                den1 = np.var(inner, axis=0)
                den2 = np.real(np.squeeze(np.expand_dims(res.conj(), 1) @ Cq_glob @ np.expand_dims(res, 2)))
                den2 = np.mean(den2, axis=0)
                rate_glob = np.log2(1 + num / (den1 + den2))
                rate_list[-1].append(rate_glob)


    mse_list = [list(i) for i in zip(*mse_list)]
    rate_list = [list(i) for i in zip(*rate_list)]
    print(mse_list)
    file_name = f'./results/' + params['model_type'] + '/' + date_time + '_ant=' + str(n_antennas) + \
                '_path=' + str(n_path) + '_ntrain=' + str(n_train_ch) + '_comp=' + str(n_components) + \
                '_pilots=' + str(n_pilots) + '_bits=' + str(n_bits) + '_0mean=' + str(params['zero_mean_gmm']) + \
                '_sums=' + str(n_summands_or_proba) + f'_ptype={pilot_type}_' \
                f'qtype={quantizer_type}_{cov_type}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)
    if eval_rate:
        file_name = f'./results/' + params['model_type'] + '/' + date_time + '_ant=' + str(n_antennas) + \
                    '_path=' + str(n_path) + '_ntrain=' + str(n_train_ch) + '_comp=' + str(n_components) + \
                    '_pilots=' + str(n_pilots) + '_bits=' + str(n_bits) + '_0mean=' + str(params['zero_mean_gmm']) + \
                    '_sums=' + str(n_summands_or_proba) + f'_ptype={pilot_type}_' \
                    f'qtype={quantizer_type}_{cov_type}_rate.csv'
        with open(file_name, 'w') as myfile:
            wr = csv.writer(myfile, lineterminator='\n')
            wr.writerows(rate_list)