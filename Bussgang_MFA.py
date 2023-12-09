import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import multiprocessing as mp
from modules.SCM3GPP.SCMMulti import SCMMulti
from modules.mofa_cplx_bussgang import Mofa
import datetime
import csv
import modules.utils as ut
from copy import deepcopy
import joblib
from modules.uniform_quantizer import get_Bussgang_matrix, get_Cr


def mp_mfa(obj, *args):
    return obj.estimate_from_y(*args)

def mp_gmm_LS(obj, *args):
    return obj.estimate_from_y_LS(*args)

if __name__ == "__main__":
    n_processes = int(mp.cpu_count() / 2)
    print('Uses ' + str(n_processes) + ' processes')
    # prepare multiprocessing
    pool = mp.Pool(processes=n_processes)

    n_antennas = 64  # BS antennas
    n_components = 64  # MFA components
    n_summands_or_proba = 'all'  # Number of MFA LMMSE that should be evaluated
    n_path = 1  # Number of propagation clusters of the 3GPP channel model
    n_pilots = 1  # Number of pilots
    n_bits = 1  # Number of quantization bits
    pilot_type = 'angle_amp'  # Pilot type {'angle', 'angle_amp', 'rand', 'ones'}
    quantizer_type = 'uniform'  # Quantizer type {'uniform', 'lloyd'}
    snrs = [-10, -5, 0, 5, 10, 15, 20]  # SNR range to be evaluated

    latent_dim = int(np.clip(n_antennas // 4, 1, np.inf)) # MFA latent dimensions
    PPCA = True # diagonal covs being scaled identities
    lock_psis = False # same diagonal covs for each component
    eval_rate = True # True if the rate lower bound should be evaluated in addition to the MSE

    params = dict()
    params['n_antennas'] = n_antennas
    params['n_comp'] = n_components
    params['n_bits'] = n_bits
    params['n_path'] = n_path
    params['quantizer_type'] = quantizer_type
    params['PPCA'] = PPCA
    params['lock_psis'] = lock_psis
    params['zero_mean_mfa'] = True

    n_channels = 110_000
    n_train_ch = 100_000 # training data
    n_val_ch = 10_000 # validation data

    mse_list = list()
    snrs_ = snrs.copy()
    snrs_.insert(0, 'SNR')
    mse_list.append(snrs_)

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
        channels = data[0]  # channel data
        toep = data[1]  # vectors to create the genie-covariances
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

    #print(toep_val.shape[0])
    params['n_pilots'] = n_pilots
    params['n_train'] = n_train_ch
    params['n_val'] = n_val_ch
    channels_train = channels[:n_train_ch]
    channels_val = channels[n_train_ch:n_train_ch+n_val_ch]

    #get pilot matrix
    A = ut.get_pilot_matrix(n_antennas, n_pilots, n_bits, pilot_type=pilot_type)

    #get quantizer
    quantizer = ut.get_quantizer(snrs, n_bits, quantizer_type=quantizer_type)

    # compute global cov
    cov = np.zeros([n_antennas, n_antennas], dtype=complex)
    for i in range(n_train_ch):
        cov = cov + np.expand_dims(channels_train[i, :], 1) @ np.expand_dims(channels_train[i, :].conj(), 0)
    cov = cov / n_train_ch

    #fit MFA model once and store it
    file_name_mfa = f'results/saves/trained_mfa_ant={n_antennas}_comp={n_components}_model={params["model_type"]}' \
                    f'_paths={params["n_path"]}_ntrain={n_train_ch}_latent={latent_dim}_PPCA={PPCA}_lockpsi=' \
                    f'{lock_psis}_zeromean={params["zero_mean_mfa"]}.sav'
    try:
        mfa_est = joblib.load(file_name_mfa)
        print('Loading trained mfa successful.')
    except FileNotFoundError:
        if (not (lock_psis or PPCA)) or params['zero_mean_mfa']:
            # prevent numerical instabilities by avoiding zero-responsibilities
            rs_clip = 1e-3
        else:
            rs_clip = 0.0
        print('Fit mfa model...')
        mfa_est = Mofa(
            n_components=n_components,
            latent_dim=latent_dim,
            PPCA=PPCA,
            lock_psis=lock_psis,
            rs_clip=rs_clip,
            max_condition_number=1.e6,
            maxiter=100,
            verbose=False,
        )
        mfa_est.fit(channels_train, params['zero_mean_mfa'])
        print('done.')

    params['n_summands_or_proba'] = n_summands_or_proba
    if eval_rate:
        mse_list.append(['mfa_rstat'])
    mse_list.append(['blmmse_mfa'])
    mfa_copy = deepcopy(mfa_est)
    mfa_list = list()
    for snr in snrs:
        #r_train = ut.get_observation_nbit(channels_train, snr, A, n_bits, quantizer[snr][0], quantizer[snr][1])
        r_val = ut.get_observation_nbit(channels_val, snr, A, n_bits, quantizer[snr][0], quantizer[snr][1])
        mfa_list.append([mfa_copy, r_val, snr, A, n_summands_or_proba, n_bits,
                    quantizer_type, quantizer[snr]])
    res_mfa_blmmse = pool.starmap(mp_mfa, mfa_list)
    for it, res in enumerate(res_mfa_blmmse):
        mse_act = np.sum(np.abs(res - channels_val)**2) / channels_val.size
        mse_list[-1].append(mse_act)
        # compute achievable rate lower bound
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
            rate_glob2 = np.log2(1 + num / (den1 + den2))
            mse_list[-2].append(rate_glob2)

    # print and save results
    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    file_name = f'./results/{params["model_type"]}/{date_time}_ant={n_antennas}_path={n_path}_train=' \
                f'{n_train_ch}_comp={n_components}_pil={n_pilots}_bits={n_bits}_sums={n_summands_or_proba}' \
                f'_L={latent_dim}_PPCA={PPCA}_lockpsi={lock_psis}_ptype={pilot_type}_qtype={quantizer_type}_' \
                f'0mean={params["zero_mean_mfa"]}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)
