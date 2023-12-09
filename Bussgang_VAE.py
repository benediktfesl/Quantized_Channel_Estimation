import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
from modules.SCM3GPP.SCMMulti import SCMMulti
from datetime import datetime
import csv
import h5py
import argparse
import modules.utils as ut
import torch
from estimators.vae import VAE_nbit
from matplotlib import pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', '-g',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--vae', '-v',
        type=int,
        default=0
    )
    parargs = parser.parse_args()
    vae_modes = ['genie', 'noisy', 'real']
    vae_mode = vae_modes[parargs.vae]

    # There are three VAE variants implemented: 'genie', 'noisy', 'real'
    # 'genie': Encoder input is the TRUE channel without noise and unquantized
    # 'noisy': Encoder input is a quantized pilot observation, training is with perfect CSI data
    # 'real': Encoder input is a quantized pilot observation, training is solely with quantized data

    n_antennas = 64 # Number of BS antennas
    n_path = 3 # Number of propagation clusters of the 3GPP channel model
    n_pilots = 1 # Number of pilots
    n_bits = 2 # Number of quantization bits
    pilot_type = 'angle_amp' # Pilot type {'angle', 'angle_amp', 'rand', 'ones'}
    quantizer_type = 'uniform' # Quantizer type {'uniform', 'lloyd'}
    snrs = [-10, -5, 0, 5, 10, 15, 20] # SNR range to be evaluated

    n_channels = 120_000
    n_train_ch = 100_000
    n_test_ch = 1_000
    n_val_ch = 10_000

    train_vae = True

    date_time_now = datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    # Some hyperparameters are chosen random to perform a random search
    params = {
        'sim_id': date_time,
        'n_antennas': n_antennas,
        'n_pilots': n_pilots,
        'pilot_type': pilot_type,
        'n_bits': n_bits,
        'snrs': snrs,
        'n_paths': n_path,
        'vae_mode': vae_mode,
        'n_train': n_train_ch,
        'n_test': n_test_ch,
        'n_val': n_val_ch,
        'epochs': 500,
        'quantizer_type': quantizer_type,
        'file_vae': '',
        'apply_batchnorm': False, #np.random.choice([True, False]),
        'lr': ut.rand_exp(1e-5, 1e-3)[0], #5*1e-4,
        'batch_size': np.random.randint(100, 300),
        'n_layers': 4, #np.random.randint(3, 6),
        'latent_dim': int(np.clip(n_antennas // 4, 1, np.inf)), #np.random.randint(8, 40),
        'zeromean': True, #np.random.choice([True, False]),
        'fft_pre': True, #np.random.choice([True, False]),
        'conv_vae': False,
        'filters_max': np.random.choice([32, 48, 64, 96, 128, 156, 256]),
        'n_pilot_convs': max(0, n_pilots // 2),
        'eval_rate': True,
        'snr_scale': False,
        'snr_scale_fac': np.random.uniform(0, 1),
    }
    if params['vae_mode'] == 'real':
        params['fft_pre'] = True

    mse_list = list()
    snrs_ = snrs.copy()
    snrs_.insert(0, 'SNR')
    mse_list.append(snrs_)

    date_time_now = datetime.now()
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

    params['n_pilots'] = n_pilots
    params['n_train'] = n_train_ch
    params['n_test'] = n_test_ch
    params['n_val'] = n_val_ch
    channels_train = channels[:n_train_ch]
    channels_test = channels[n_train_ch:n_train_ch+n_test_ch]
    channels_val = channels[n_train_ch+n_test_ch:n_train_ch+n_test_ch+n_val_ch]

    if parargs.gpu >= 0:
        print('Run on GPU ' + str(parargs.gpu) + '.')
        device = torch.device('cuda:' + str(parargs.gpu))
    else:
        print('Run on CPU.')
        device = torch.device('cpu')
    params['device'] = device

    print("\n".join("{!r}: {!r},".format(k, v) for k, v in params.items()))
    print('=' * 20)

    # get pilot matrix
    A = ut.get_pilot_matrix(n_antennas, n_pilots, n_bits, pilot_type=pilot_type)
    params['A'] = A

    # get quantizer
    quantizer = ut.get_quantizer(snrs, n_bits, quantizer_type=quantizer_type)
    params['quantizer'] = quantizer

    # initialize VAE estimator
    vae_est = VAE_nbit(params=params)

    # train VAE
    if train_vae:
        losses_all, losses_all_test = vae_est.train(channels_train, channels_test, snrs)

    # eval VAE
    mse_list.append([f'vae_{params["vae_mode"]}'])
    mse_list.append([f'vae_{params["vae_mode"]}_rstat'])
    for snr in snrs:
        r = ut.get_observation_nbit(channels_val, snr, A, n_bits, quantizer[snr][0], quantizer[snr][1])
        mse, rate, params = vae_est.eval(channels_val, r, snr, channels_train)
        mse_list[-2].append(mse)
        mse_list[-1].append(rate)

    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    file_name = f'./results/vae/{date_time}_vae{vae_mode}_{params["model_type"]}_path={params["n_paths"]}_ant=' \
                f'{n_antennas}_bits={n_bits}_train={n_train_ch}_pilot={n_pilots}_qtype={quantizer_type}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)

    #save params
    params['quantizer'] = None
    params['A'] = None
    file_name = f'./results/vae/{date_time}_vae{vae_mode}_{params["model_type"]}_path={params["n_paths"]}_ant=' \
                f'{n_antennas}_bits={n_bits}_train={n_train_ch}_pilot={n_pilots}_qtype={quantizer_type}_params.csv'
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
           writer.writerow([key, value])

    if train_vae:
        #save pyplot
        file_name = f'./results/vae/{date_time}_vae{vae_mode}_{params["model_type"]}_path={params["n_paths"]}_ant=' \
                    f'{n_antennas}_bits={n_bits}_train={n_train_ch}_pilot={n_pilots}_qtype={quantizer_type}_loss.png'
        plt.plot(range(1, params['epochs']+1), losses_all, label='train-loss')
        plt.plot(range(1, params['epochs']+1), losses_all_test, label='val-loss')
        plt.legend(['train-loss', 'val-loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(file_name)