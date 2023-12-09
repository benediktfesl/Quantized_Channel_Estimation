import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from modules import utils as ut
from modules import uniform_quantizer as quant_uni
from modules import lloyd_max_quantizer as quant_lloyd


class VAE_nbit:
    def __init__(self, params):
        self.params = params

    def train(self, h_train, h_test=None, snr_list=None):
        torch.set_num_threads(3)
        vae_est = DNN_VAE(self.params)
        if self.params['snr_scale']:
            snr_scale = np.linspace(1, self.params['snr_scale_fac'], len(self.params['snrs']))
            snr_scale /= np.sum(snr_scale)
        else:
            snr_scale = None

        pytorch_total_params = sum(p.numel() for p in vae_est.parameters())
        pytorch_total_params_train = sum(p.numel() for p in vae_est.parameters() if p.requires_grad)
        #print(vae_est)
        if self.params['vae_mode'] == 'genie':
            h_train_fft = np.fft.fft(h_train, axis=1) / np.sqrt(h_train.shape[1])
            h_train_fft = ut.cplx2real(h_train_fft, axis=1)
            h_train = torch.as_tensor(h_train_fft, device=self.params['device']).float()
            dataset_train = TensorDataset(h_train)
            dataloader_train = DataLoader(dataset_train, batch_size=self.params['batch_size'], shuffle=True,
                                          drop_last=True)
            if h_test is not None:
                h_test_fft = np.fft.fft(h_test, axis=1) / np.sqrt(h_test.shape[1])
                h_test_fft = ut.cplx2real(h_test_fft, axis=1)
                h_test = torch.as_tensor(h_test_fft, device=self.params['device']).float()
                dataset_test = TensorDataset(h_test)
                dataloader_test = DataLoader(dataset_test, batch_size=self.params['batch_size'], shuffle=True,
                                              drop_last=True)
        elif self.params['vae_mode'] == 'noisy':
            h_train_fft = np.fft.fft(h_train, axis=1) / np.sqrt(h_train.shape[1])
            h_train_fft = ut.cplx2real(h_train_fft, axis=1)
            h_train_fft = torch.as_tensor(h_train_fft, device=self.params['device']).float()
            if h_test is not None:
                r_test, _ = ut.get_observation_nbit_randSNR(h_test, snr_list, self.params['A'], self.params['n_bits'],
                                                             self.params['quantizer'], snr_scaling=snr_scale)
                r_test = np.reshape(r_test, (-1, self.params['n_antennas'], self.params['n_pilots']), 'F')
                r_test = np.transpose(r_test, [0, 2, 1])
                if self.params['fft_pre']:
                    r_test = np.fft.fft(r_test, axis=-1) / np.sqrt(r_test.shape[-1])
                r_test = ut.cplx2real(r_test, axis=-1)
                r_test = torch.as_tensor(r_test, device=self.params['device']).float()
                h_test_fft = np.fft.fft(h_test, axis=1) / np.sqrt(h_test.shape[1])
                h_test_fft = ut.cplx2real(h_test_fft, axis=1)
                h_test = torch.as_tensor(h_test_fft, device=self.params['device']).float()
                dataset_test = TensorDataset(h_test, r_test)
                dataloader_test = DataLoader(dataset_test, batch_size=self.params['batch_size'], shuffle=True,
                                              drop_last=True)
        elif self.params['vae_mode'] == 'real':
            if h_test is not None:
                r_test, snr_list_test = ut.get_observation_nbit_randSNR(h_test, snr_list, self.params['A'], self.params['n_bits'],
                                                            self.params['quantizer'], snr_scaling=snr_scale)
                if self.params['fft_pre']:
                    r_test = np.fft.fft(r_test, axis=1) / np.sqrt(r_test.shape[1])
                r_test = ut.cplx2real(r_test, axis=1)
                r_test = torch.as_tensor(r_test, device=self.params['device']).float()
                snr_list_test = torch.as_tensor(snr_list_test, device=self.params['device']).float()
                dataset_test = TensorDataset(r_test, snr_list_test)
                dataloader_test = DataLoader(dataset_test, batch_size=self.params['batch_size'], shuffle=True,
                                             drop_last=True)
        else:
            NotImplementedError('Choose a valid VAE-option.')
        #dataloader_train = DataLoader(dataset_train, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
        opti = optim.Adam(vae_est.parameters(), lr=self.params['lr'])
        filename_vae = f'results/vae/saves/VAE{self.params["vae_mode"]}_{self.params["sim_id"]}_' \
                       f'{self.params["model_type"]}_paths={self.params["n_paths"]}_ant=' \
                       f'{self.params["n_antennas"]}_ntrain={self.params["n_train"]}.pt'
        self.params['file_vae'] = filename_vae
        losses_all = list()
        losses_all_test = list()

        h_train = torch.as_tensor(h_train, dtype=torch.complex128, device=self.params['device'])
        A_torch = torch.as_tensor(self.params['A'], dtype=h_train.dtype, device=self.params['device'])

        for epoch in range(self.params['epochs']):
            loss_epoch = list()
            if self.params['vae_mode'] == 'noisy':
                r_train, _ = ut.get_observation_nbit_randSNR_torch_fast(h_train, snr_list, A_torch, self.params['n_bits'], self.params['quantizer'], snr_scaling=snr_scale, device=self.params['device'])
                r_train = ut.reshape_fortran(r_train, (-1, self.params['n_antennas'], self.params['n_pilots']))
                r_train = torch.transpose(r_train, 2, 1)
                if self.params['fft_pre']:
                    r_train = torch.fft.fft(r_train, axis=-1) / np.sqrt(r_train.shape[-1])
                r_train = ut.cplx2real_torch(r_train, axis=-1).float()
                dataset_train = TensorDataset(h_train_fft, r_train)
                dataloader_train = DataLoader(dataset_train, batch_size=self.params['batch_size'], shuffle=True,
                                              drop_last=True)
            elif self.params['vae_mode'] == 'real':
                r_train, snr_list_train = ut.get_observation_nbit_randSNR_torch_fast(h_train, snr_list, A_torch, self.params['n_bits'], self.params['quantizer'], snr_scaling=snr_scale, device=self.params['device'])
                r_train = torch.fft.fft(r_train, axis=1) / np.sqrt(r_train.shape[1])
                r_train = ut.cplx2real_torch(r_train, axis=1).float()
                snr_list_train = torch.as_tensor(snr_list_train, device=self.params['device']).float()
                dataset_train = TensorDataset(r_train, snr_list_train)
                dataloader_train = DataLoader(dataset_train, batch_size=self.params['batch_size'], shuffle=True,
                                              drop_last=True)
            for iter_ in range(len(dataloader_train)):
                opti.zero_grad()
                vae_est.zero_grad()
                if self.params['vae_mode'] == 'genie':
                    h_batch = next(iter(dataloader_train))[0]
                    y_batch = None
                    snr_list_loss = None
                elif self.params['vae_mode'] == 'noisy':
                    h_batch, y_batch = next(iter(dataloader_train))
                    snr_list_loss = None
                elif self.params['vae_mode'] == 'real':
                    y_batch, snr_list_loss = next(iter(dataloader_train))
                    h_batch = None
                loss = vae_est.vae_loss(h_batch, y_batch, mode=self.params['vae_mode'], snr_list=snr_list_loss)
                if np.isnan(loss.cpu().detach().numpy()) or loss.cpu().detach().numpy() > 1_000:
                    continue
                loss.backward()
                opti.step()
                loss_epoch.append(float(loss.cpu().detach().numpy()))
            if loss_epoch: #check if list is empty
                if np.isnan(np.mean(loss_epoch)):
                    continue
                losses_all.append(np.clip(np.mean(loss_epoch), -np.inf, 1_000))
                # evaluate status with unseen data
                with torch.no_grad():
                    loss_epoch_test = list()
                    for iter_ in range(len(dataloader_test)):
                        if self.params['vae_mode'] == 'genie':
                            h_batch = next(iter(dataloader_test))[0]
                            y_batch = None
                            snr_list_loss = None
                        elif self.params['vae_mode'] == 'noisy':
                            h_batch, y_batch = next(iter(dataloader_test))
                            snr_list_loss = None
                        elif self.params['vae_mode'] == 'real':
                            y_batch, snr_list_loss = next(iter(dataloader_test))
                            h_batch = None
                        loss = vae_est.vae_loss(h_batch, y_batch, mode=self.params['vae_mode'], snr_list=snr_list_loss)
                        loss_epoch_test.append(float(loss.cpu().detach().numpy()))
                losses_all_test.append(np.clip(np.mean(loss_epoch_test), -np.inf, 1_000))
                # print status after each epoch
                print(f'epoch: {epoch+1}/{self.params["epochs"]} | train-loss: {losses_all[-1]:,.1f} | val-loss: {losses_all_test[-1]:,.1f}', end="\r")
                torch.save({'model': vae_est.state_dict(),
                            'optim': opti.state_dict(),
                            'loss_all': losses_all,
                            'epoch': epoch,
                            'params': self.params},
                            filename_vae)
        return losses_all, losses_all_test


    def eval(self, h_eval, y_eval, snr, h_train):
        h_train = torch.as_tensor(h_train, dtype=torch.complex128, device=self.params['device'])
        norm_fac = h_eval.size
        # fft-transformed channels
        h_eval_fft = np.fft.fft(h_eval, axis=1) / np.sqrt(h_eval.shape[1])
        h_eval_fft = ut.cplx2real(h_eval_fft, axis=1)
        h_eval_fft = torch.as_tensor(h_eval_fft, device=self.params['device']).float()
        # fft-transformed observations
        y_eval_fft = np.reshape(y_eval, (-1, self.params['n_antennas'], self.params['n_pilots']), 'F')
        y_eval_fft = np.transpose(y_eval_fft, [0, 2, 1])
        if self.params['fft_pre']:
            y_eval_fft = np.fft.fft(y_eval_fft, axis=-1) / np.sqrt(y_eval_fft.shape[-1])
            y_eval_fft = ut.cplx2real(y_eval_fft, axis=-1)
        else:
            y_eval_fft = ut.cplx2real(y_eval_fft, axis=-1)
        y_eval_fft = torch.as_tensor(y_eval_fft, device=self.params['device']).float()
        h_eval = torch.as_tensor(h_eval, dtype=torch.complex128, device=self.params["device"])
        # cplx-valued observations
        y_eval = torch.as_tensor(y_eval, dtype=torch.complex128, device=self.params["device"])
        dataset_eval = TensorDataset(h_eval_fft, y_eval_fft, h_eval, y_eval)
        dataloader_eval = DataLoader(dataset_eval, batch_size=50, shuffle=False, drop_last=False)
        filename_vae = self.params['file_vae']
        ckpnt = torch.load(filename_vae, map_location=self.params['device'])
        self.params = assign_params(ckpnt['params'], self.params)
        vae_est = DNN_VAE(self.params)
        vae_est.load_state_dict(ckpnt['model'])
        sigma2 = 10 ** (-snr / 10)
        F = torch.fft.fft(torch.eye(h_eval.shape[1], device=self.params['device'], dtype=torch.complex128)) / np.sqrt(h_eval.shape[1])
        if self.params['eval_rate']:
            # compute rate lower bound: global
            cov = torch.zeros([self.params["n_antennas"], self.params["n_antennas"]], dtype=torch.complex128,
                              device=self.params['device'])
            for i in range(h_train.shape[0]):
                cov = cov + torch.unsqueeze(h_train[i, :], 1) @ torch.unsqueeze(h_train[i, :].conj(), 0)
            cov = cov / h_train.shape[0]
            Cy_act = cov + 10 ** (-snr / 10) * torch.eye(self.params["n_antennas"], dtype=torch.complex128, device=self.params['device'])
            Buss_glob = quant_uni.get_Bussgang_matrix_torch(snr, self.params['n_bits'], Cy_act, self.params['device'])
            Cr = quant_uni.get_Cr_torch(Cy_act, self.params['n_bits'], snr, self.params['quantizer'][snr], device=self.params['device'])
            Cq_glob = Cr - Buss_glob @ cov @ Buss_glob.conj().T
        A_torch = torch.as_tensor(self.params["A"], device=self.params['device'], dtype=torch.complex128)
        hest_all = torch.zeros_like(h_eval)
        rate = 0.0
        count = 0
        with torch.no_grad():
            mse = 0.0
            for iter_, (h_batch, y_batch, h_true_batch, y_true_batch) in enumerate(dataloader_eval):
                if self.params['vae_mode'] == 'genie':
                    mu, var = vae_est.forward_nosamp(h_batch)
                else:
                    mu, var = vae_est.forward_nosamp(y_batch)
                mu_h, mu_y, Ch, Cy_inv = convert_dec_outputs(mu, var, F, sigma2, self.params["n_antennas"],
                                            A_torch, self.params['n_bits'], self.params['device'], self.params['quantizer'][snr],
                                                             self.params['quantizer_type'], self.params['zeromean'])
                h_hat_batch = lmmse(y_true_batch, mu_h, mu_y, Ch, Cy_inv)
                mse += np.sum(np.abs(h_hat_batch.cpu().detach().numpy() - h_true_batch.cpu().detach().numpy()) ** 2)
                if self.params['eval_rate']:
                    # evaluate statistical lower bound
                    norm_fac_rate = torch.sum(torch.abs(h_hat_batch) ** 2, dim=1)
                    for i in range(h_hat_batch.shape[0]):
                        h_hat_batch[i] /= norm_fac_rate[i]
                    hest_all[count:count + h_hat_batch.shape[0]] = h_hat_batch
                    count += h_hat_batch.shape[0]
            if self.params['eval_rate']:
                inner = torch.squeeze(torch.unsqueeze(hest_all.conj(), 1) @ Buss_glob @ torch.unsqueeze(h_eval, 2))
                num = torch.abs(torch.mean(inner, dim=0)) ** 2
                den1 = torch.var(inner, dim=0)
                den2 = torch.real(torch.squeeze(torch.unsqueeze(hest_all.conj(), 1) @ Cq_glob @ torch.unsqueeze(hest_all, 2)))
                den2 = torch.mean(den2, dim=0)
                rate = torch.log2(1 + num / (den1 + den2))
        rate = float(rate.cpu().detach().numpy())
        mse /= norm_fac
        return mse, rate, self.params



class DNN_VAE(nn.Module):
    def __init__(self, params):
        super(DNN_VAE, self).__init__()
        self.params = params
        self.device = params['device']
        self.neurons_enc = np.linspace(2*self.params['n_antennas'], 2*self.params['latent_dim'], self.params['n_layers'] + 1, dtype=int)
        self.enc = nn.Sequential().to(self.device)
        self.filters = np.linspace(2, self.params['filters_max'], self.params['n_layers'], dtype=int)
        self.filters_rev = np.linspace(6, self.params['filters_max'], self.params['n_layers'] // 2 + 1, dtype=int)
        self.pilot_convs = np.linspace(self.params['n_pilots'], 1, self.params['n_pilot_convs'] + 1, dtype=int)
        self.pre_pilot = nn.Sequential().to(self.device)
        for i in range(self.params['n_pilot_convs']):
            self.pre_pilot.add_module(f'conv{i}', nn.Conv1d(self.pilot_convs[i], self.pilot_convs[i+1], 1, device=self.device))
            self.pre_pilot.add_module(f'act_pre{i}', nn.ReLU())
        for i in range(self.params['n_layers']):
            self.enc.add_module(f'linear{i}', nn.Linear(self.neurons_enc[i], self.neurons_enc[i+1], device=self.device))
            if i < self.params['n_layers'] - 1:
                if self.params['apply_batchnorm']:
                    self.enc.add_module(f'batch{i}', nn.BatchNorm1d(self.neurons_enc[i+1], device=self.device))
                self.enc.add_module(f'act{i}', nn.ReLU())
        #self.enc_mu = nn.Linear(self.neurons_enc[-1], self.params['latent_dim'], device=self.device)
        #self.enc_var = nn.Linear(self.neurons_enc[-1], self.params['latent_dim'], device=self.device)

        if self.params['zeromean']:
            self.neurons_dec = np.linspace(self.params['latent_dim'], self.params['n_antennas'],
                                       self.params['n_layers'] + 1, dtype=int)
        else:
            self.neurons_dec = np.linspace(self.params['latent_dim'], 3 * self.params['n_antennas'],
                                       self.params['n_layers'] + 1, dtype=int)
        self.dec = nn.Sequential().to(self.device)
        for i in range(self.params['n_layers']):
            self.dec.add_module(f'linear{i}', nn.Linear(self.neurons_dec[i], self.neurons_dec[i+1], device=self.device))
            if i < self.params['n_layers'] - 1:
                if self.params['apply_batchnorm']:
                    self.dec.add_module(f'batch{i}', nn.BatchNorm1d(self.neurons_dec[i+1], device=self.device))
                self.dec.add_module(f'act{i}', nn.ReLU())
        #self.dec_mu = nn.Linear(self.neurons_dec[-1], 2*self.params['n_antennas'], device=self.device)
        #self.dec_var = nn.Linear(self.neurons_dec[-1], self.params['n_antennas'], device=self.device)


    def forward(self, x):
        if self.params['vae_mode'] is not 'genie':
            x = torch.squeeze(self.pre_pilot(x))
        x = self.enc(x)
        #mu_enc = self.enc_mu(x)
        #var_enc = self.enc_var(x)
        mu_enc = x[:, :(x.shape[-1] // 2)]
        var_enc = x[:, (x.shape[-1] // 2):]
        epsilon = torch.randn(mu_enc.shape).to(self.device)
        z = mu_enc + torch.exp(var_enc) * epsilon
        if self.params['zeromean']:
            mu_dec = None
            var_dec = self.dec(z)
        else:
            x = self.dec(z)
            mu_dec = x[:, :int(2*x.shape[-1] / 3)]
            var_dec = x[:, int(2*x.shape[-1] / 3):]
        #mu_dec = self.dec_mu(x)
        #var_dec = self.dec_var(x)
        return mu_enc, var_enc, mu_dec, var_dec


    def forward_nosamp(self, x):
        #directly forward latent mean for evaluation of estimator
        if self.params['vae_mode'] is not 'genie':
            x = torch.squeeze(self.pre_pilot(x))
        x = self.enc(x)
        mu_enc = x[:, :(x.shape[-1] // 2)]
        if self.params['zeromean']:
            mu_dec = None
            var_dec = self.dec(mu_enc)
        else:
            x = self.dec(mu_enc)
            mu_dec = x[:, :int(2 * x.shape[-1] / 3)]
            var_dec = x[:, int(2 * x.shape[-1] / 3):]
        #mu_dec = self.dec_mu(x)
        #var_dec = self.dec_var(x)
        return mu_dec, var_dec


    def vae_loss(self, data_h, data_y, mode, snr_list=None):
        if mode == 'genie':
            mu_enc, log_var_enc, mu_dec, log_var_dec = self.forward(data_h)
        else:
            mu_enc, log_var_enc, mu_dec, log_var_dec = self.forward(data_y)

        # w_delta^H diag(lambda) w_delta
        # to compute w_delta^H diag(lambda) w_delta, we essentially have a multiplication of the form
        # (a-jb) * c * (a+jb) = a*c*a + b*c*b
        # where c is used twice - once for the real and once for the imaginary parts
        if mode == 'real':
            # sum over log(lambda)
            sigma2 = 10 ** (-snr_list / 10)
            cy = torch.exp(-log_var_dec) + sigma2.unsqueeze(1).repeat(1, log_var_dec.shape[1])
            if self.params['n_bits'] != np.inf:
                cy_diag = torch.mean(cy, dim=1).unsqueeze(1).repeat(1, log_var_dec.shape[1])
                if self.params['quantizer_type'] == 'uniform':
                    Buss_fac = quant_uni.get_Bussgang_matrix_diag_fast(snr_list, self.params['n_bits'], cy_diag[:,0])
                else:
                    raise NotImplementedError
                Buss_fac = torch.clamp(torch.pow(Buss_fac, 2).unsqueeze(1).repeat(1, log_var_dec.shape[1]), 0, 1)
                cy = Buss_fac * cy + (torch.ones_like(Buss_fac) - Buss_fac) * cy_diag
            loss = torch.sum(-torch.log(cy), dim=1)
            if self.params['zeromean']:
                w_delta = torch.clone(data_y)
            else:
                w_delta = data_y - mu_dec
            #test = sigma2.unsqueeze(1).repeat(1, log_var_dec.shape[1])
            #c = 1 / (torch.exp(-log_var_dec) + sigma2.unsqueeze(1).repeat(1, log_var_dec.shape[1]))
            diag_lambda_w_delta = torch.cat([1/cy, 1/cy], dim=1) * w_delta
            loss -= torch.einsum('ij,ij->i', w_delta, diag_lambda_w_delta)
        else:
            # sum over log(lambda)
            loss = torch.sum(log_var_dec, dim=1)
            if self.params['zeromean']:
                #w_delta = torch.clone(data_h)
                diag_lambda_w_delta = torch.exp(torch.cat([log_var_dec, log_var_dec], dim=1)) * data_h
                loss -= torch.einsum('ij,ij->i', data_h, diag_lambda_w_delta)
            else:
                w_delta = data_h - mu_dec
                diag_lambda_w_delta = torch.exp(torch.cat([log_var_dec, log_var_dec], dim=1)) * w_delta
                loss -= torch.einsum('ij,ij->i', w_delta, diag_lambda_w_delta)

        # sum over log(sigma)
        loss += torch.sum(log_var_enc, dim=1)

        # sum over mu_h^2
        loss -= 0.5 * torch.sum(mu_enc**2, dim=1)

        # sum over sigma_h^2, where 2*log(sigma) = log(sigma^2) is used
        loss -= 0.5 * torch.sum(torch.exp(2*log_var_enc), dim=1)

        # minus because loss has to be maximized
        return -loss.mean()


def lmmse(y, mu_h, mu_y, Ch, Cy_inv):
    return mu_h + torch.squeeze(Ch @ Cy_inv @ (y - mu_y)[:, :, None])


def lmmse_np(y, mu_h, mu_y, Ch, Cy_inv):
    return mu_h + np.squeeze(Ch @ Cy_inv @ (y - mu_y)[:, :, None])


def convert_dec_outputs(mu, var, F, sigma2, n_antennas, A, n_bits, device, quantizer, quantizer_type='uniform', zeromean=False):
    #snr = -10*np.log10(sigma2)
    if not zeromean:
        #mu = mu.cpu().detach().numpy()
        mu = ut.real2cplx_torch(mu, axis=1)
        mu_h = mu @ torch.conj(F)
    else:
        mu_h = torch.zeros([var.shape[0], n_antennas], device=device, dtype=torch.complex128)
    var = torch.clip(torch.exp(-var), 1e-12, np.inf)
    diag_c = torch.eye(n_antennas, dtype=torch.complex128, device=device) * var[:, np.newaxis, :]
    Ch = torch.conj(F).T @ diag_c @ F
    #F_eff = A @ torch.conj(F).T
    Cy = A @ Ch @ torch.conj(A).T + sigma2 * torch.eye(A.shape[0], dtype=torch.complex128, device=device)
    #var_y = var + sigma2
    if n_bits == np.inf:
        Cy_inv = torch.linalg.pinv(Cy, hermitian=True)
        #Cy_inv = torch.eye(n_antennas, dtype=torch.complex128, device=device) * (1 / var_y[:, np.newaxis, :])
        #Cy_inv = F_eff @ Cy_inv @ torch.conj(F_eff).T
        Ch = Ch @ torch.conj(A).T
        mu_y = torch.squeeze(A @ mu_h[:, :, None])
    elif n_bits == 1:
        #Cy = F_eff @ (torch.eye(n_antennas, dtype=torch.complex128, device=device) * var_y[:, np.newaxis, :]) @ torch.conj(F_eff).T
        #Psi_12 = np.real(np.diag(1 / np.sqrt(np.diag(Cy))))
        Psi_12 = torch.zeros_like(Cy, dtype=torch.double)
        for i in range(Cy.shape[0]):
            Psi_12[i] = torch.real(torch.diag(1 / torch.sqrt(torch.diag(Cy[i]))))
        # if A_eff is None:
        A_eff = torch.sqrt(2 / torch.tensor(np.pi)) * torch.matmul(Psi_12.type(torch.complex128), A)
        inner_real = torch.real(torch.matmul(Psi_12, torch.real(Cy)) @ Psi_12)
        inner_imag = torch.real(torch.matmul(Psi_12, torch.imag(Cy)) @ Psi_12)
        inner_real[inner_real > 1] = 1.0
        inner_imag[inner_imag > 1] = 1.0
        inner_real[inner_real < -1] = -1.0
        inner_imag[inner_imag < -1] = -1.0
        Cr = (2 / torch.tensor(np.pi) * (torch.asin(inner_real) + 1j * torch.asin(inner_imag))).type(torch.complex128)
        mu_y = torch.squeeze(torch.matmul(A_eff, mu_h[:, :, None]))
        Cy_inv = torch.linalg.pinv(Cr, hermitian=True)
        Ch = torch.matmul(Ch, torch.transpose(A_eff.conj(), 1, 2))
    else:
        #Cy = torch.matmul(torch.matmul(F_eff, torch.eye(n_antennas, dtype=torch.complex128, device=device)) * var_y[:, None, :],
        #                  torch.transpose(F_eff.conj(), 0, 1))
        snr = -10 * np.log10(sigma2)
        Cr = torch.zeros_like(Cy)
        A_buss = torch.zeros_like(Cy)
        for i in range(Cy.shape[0]):
            if quantizer_type == 'uniform':
                A_buss[i] = quant_uni.get_Bussgang_matrix_torch(snr_dB=snr, n_bits=n_bits, Cy=Cy[i], device=device)
            elif quantizer_type == 'lloyd':
                A_buss[i] = quant_lloyd.get_Bussgang_matrix(n_bits=n_bits, Cy=Cy[i], quantizer=quantizer)
            beta = torch.clip(torch.real(torch.mean(torch.diag(A_buss[i]))), 0, 1).type(torch.complex128)
            Cr[i] = beta ** 2 * Cy[i] + (1 - beta ** 2) * torch.diag(torch.diag(Cy[i]))
        A_eff = torch.matmul(A_buss, A)
        mu_y = torch.squeeze(torch.matmul(torch.matmul(A_buss, A), mu_h[:, :, None]))
        Ch = torch.matmul(Ch, torch.transpose(A_eff.conj(), 1, 2))
        Cy_inv = torch.linalg.pinv(Cr, hermitian=True)
    return mu_h, mu_y, Ch, Cy_inv


def convert_dec_outputs_np(mu, var, F, sigma2, n_antennas, A, n_bits, quantizer, quantizer_type='uniform', zeromean=False):
    #snr = -10*np.log10(sigma2)
    if not zeromean:
        mu = mu.cpu().detach().numpy()
        mu = ut.real2cplx_torch(mu, axis=1)
        mu_h = mu @ F.conj()
    else:
        mu_h = np.zeros([var.shape[0], n_antennas], dtype=complex)
    var = np.clip(np.exp(-var.cpu().detach().numpy()), 1e-12, np.inf)
    diag_c = np.eye(n_antennas, dtype=complex) * var[:, np.newaxis, :]
    Ch = F.conj().T @ diag_c @ F
    F_eff = A @ F.conj().T
    var_y = var + sigma2
    if n_bits == np.inf:
        Cy_inv = np.eye(n_antennas, dtype=complex) * (1 / var_y[:, np.newaxis, :])
        Cy_inv = F_eff @ Cy_inv @ F_eff.conj().T
        Ch = Ch @ A.conj().T
        mu_y = np.squeeze(A @ mu_h[:, :, None])
    elif n_bits == 1:
        Cy = F_eff @ (np.eye(n_antennas, dtype=complex) * var_y[:, np.newaxis, :]) @ F_eff.conj().T
        #Psi_12 = np.real(np.diag(1 / np.sqrt(np.diag(Cy))))
        Psi_12 = np.zeros_like(Cy)
        for i in range(Cy.shape[0]):
            Psi_12[i] = np.real(np.diag(1 / np.sqrt(np.diag(Cy[i]))))
        # if A_eff is None:
        A_eff = np.sqrt(2 / np.pi) * Psi_12 @ A
        inner_real = np.real(Psi_12 @ np.real(Cy) @ Psi_12)
        inner_imag = np.real(Psi_12 @ np.imag(Cy) @ Psi_12)
        inner_real[inner_real > 1] = 1.0
        inner_imag[inner_imag > 1] = 1.0
        inner_real[inner_real < -1] = -1.0
        inner_imag[inner_imag < -1] = -1.0
        Cr = 2 / np.pi * (np.arcsin(inner_real) + 1j * np.arcsin(inner_imag))
        # update output for Bussgang LMMSE
        mu_y = np.squeeze(A_eff @ mu_h[:, :, None])
        Cy_inv = np.linalg.pinv(Cr)
        Ch = Ch @ np.transpose(A_eff.conj(), [0,2,1])
    else:
        Cy = F_eff @ (np.eye(n_antennas, dtype=complex) * var_y[:, np.newaxis, :]) @ F_eff.conj().T
        #Cy_inv = np.zeros_like(Cy)
        snr = -10 * np.log10(sigma2)
        Cr = np.zeros_like(Cy)
        #mu_y = np.zeros([Cy.shape[0], Cy.shape[-1]], dtype=Cy.dtype)
        A_buss = np.zeros_like(Cy)
        for i in range(Cy.shape[0]):
            if quantizer_type == 'uniform':
                A_buss[i] = quant_uni.get_Bussgang_matrix(snr_dB=snr, n_bits=n_bits, Cy=Cy[i])
            elif quantizer_type == 'lloyd':
                A_buss[i] = quant_lloyd.get_Bussgang_matrix(n_bits=n_bits, Cy=Cy[i], quantizer=quantizer)
            beta = np.mean(np.diag(A_buss[i]))
            Cr[i] = beta ** 2 * Cy[i] + (1 - beta ** 2) * np.diag(np.diag(Cy[i]))#np.eye(Cy.shape[-1])
            #eigvals, Q = np.linalg.eigh(Cy[i])
            #eigvals[eigvals < 1e-6] = 1e-6
            #Cr[i] = Q @ np.diag(eigvals) @ Q.conj().T
        A_eff = A_buss @ A
        mu_y = np.squeeze(A_buss @ A @ mu_h[:, :, None])
        Ch = Ch @ np.transpose(A_eff.conj(), [0, 2, 1])
        Cy_inv = np.linalg.pinv(Cr)
    return mu_h, mu_y, Ch, Cy_inv


def assign_params(params_in, params_out):
    # update hyperparameters
    params_out['n_layers'] = params_in['n_layers']
    params_out['lr'] = params_in['lr']
    params_out['batch_size'] = params_in['batch_size']
    params_out['apply_batchnorm'] = params_in['apply_batchnorm']
    params_out['latent_dim'] = params_in['latent_dim']
    return params_out
