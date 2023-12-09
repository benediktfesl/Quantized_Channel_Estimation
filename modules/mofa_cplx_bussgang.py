import numpy as np
import scipy.linalg
from scipy.linalg import inv
from sklearn import cluster
from modules import utils as ut
from modules import uniform_quantizer as quant_uni
from modules import lloyd_max_quantizer as quant_lloyd


class Mofa(object):
    """
    Mixture of Factor Analyzers using batches with moderate size in EM to avoid memory conflicts.

    calling arguments:

    [ROSS DOCUMENT HERE]

    internal variables:

    `K`:           Number of components
    `M`:           Latent dimensionality
    `D`:           Data dimensionality
    `N`:           Number of data points
    `data`:        (N,D) array of observations
    `latents`:     (K,M,N) array of latent variables
    `latent_covs`: (K,M,M,N) array of latent covariances
    `lambdas`:     (K,M,D) array of loadings
    `psis`:        (K,D) array of diagonal variance values
    `rs`:          (K,N) array of responsibilities
    `amps`:        (K) array of component amplitudes
    maxiter:
        The maximum number of iterations to try.
    tol:
        The tolerance on the relative change in the loss function that
        controls convergence.
    verbose:
        Print all the messages?
    """

    def __init__(self,
                 n_components,
                 latent_dim,
                 PPCA=False,
                 lock_psis=False,
                 rs_clip=0.0,
                 max_condition_number=1.e6,
                 maxiter=100,
                 tol=1e-6,
                 verbose=True,
                 ):

        # required
        self.n_components = n_components
        self.M = latent_dim

        # options
        self.PPCA = PPCA
        self.lock_psis = lock_psis
        self.rs_clip = rs_clip
        self.L_all = list()
        #self.init_ppca = init_ppca
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose
        self.max_condition_number = float(max_condition_number)
        assert rs_clip >= 0.0

        self.N = None
        self.D = None
        self.betas = None
        self.latents = None
        self.latent_covs = None
        self.kmeans_rs = None
        self.rs = None
        self.logLs = None
        self.batch_size = None

        # member variables used for calculating e.g. responsibilities
        self._means = None
        self._lambdas = None
        self._covs = None
        self._inv_covs = None
        self._psis = None
        # fixed variables after training. Not used for calculating e.g. responsibilities
        self.means = None
        self.lambdas = None
        self.covs = None
        self.inv_covs = None
        self.psis = None
        self.means_sub = None
        self.zero_mean = False


    def fit(self, data, zero_mean=False):
        self.zero_mean = zero_mean
        self.N = data.shape[0]
        self.D = data.shape[1]
        self.rs = np.zeros((self.n_components, self.N))
        self._covs = np.zeros((self.n_components, self.D, self.D), dtype=complex)
        self._inv_covs = np.zeros_like(self._covs)

        # initialize
        self._initialize(data)
        # run em algorithm
        self.run_em(data)
        # delete unnecessary memory
        del self.latents, self.latent_covs, self.rs, self.kmeans_rs, self.betas, self.logLs
        # store fixed parameters
        self.means = self._means.copy()
        self.covs = self._covs.copy()
        self.inv_covs = self._inv_covs.copy()
        self.psis = self._psis.copy()
        self.lambdas = self._lambdas.copy()



    def estimate_from_y(self, y, snr_dB, A=None, n_summands_or_proba=1, n_bits=1,
                        quantizer_type='uniform', quantizer=None):
        if A is None:
            A = np.eye(self.D, dtype=y.dtype)

        Cy_invs, A_eff = self._prepare_for_prediction(y, A, snr_dB, n_bits, quantizer_type, quantizer)

        h_est = np.zeros([y.shape[0], A.shape[-1]], dtype=y.dtype)
        if isinstance(n_summands_or_proba, int):
            # n_summands_or_proba represents a number of summands

            if n_summands_or_proba == 1:
                # use predicted label to choose the channel covariance matrix
                labels = self.predict_proba_max(y)
                for yi in range(y.shape[0]):
                    h_est[yi] = self._lmmse(y[yi], A_eff, labels[yi], Cy_invs[labels[yi]])
            else:
                # use predicted probabilites to compute weighted sum of estimators
                proba = self.predict_proba(y)
                for yi in range(y.shape[0]):
                    # indices for probabilites in descending order
                    idx_sort = np.argsort(proba[yi, :])[::-1]
                    for argproba in idx_sort[:n_summands_or_proba]:
                        h_est[yi, :] += proba[yi, argproba] * self._lmmse(y[yi], A_eff, argproba, Cy_invs[argproba])
                    h_est[yi, :] /= np.sum(proba[yi, idx_sort[:n_summands_or_proba]])
        elif n_summands_or_proba == 'all':
            # use all predicted probabilities to compute weighted sum of estimators
            proba = self.predict_proba(y)
            for yi in range(y.shape[0]):
                for argproba in range(proba.shape[1]):
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse(y[yi], A_eff, argproba, Cy_invs[argproba])
        else:
            # n_summands_or_proba represents a probability
            # use predicted probabilites to compute weighted sum of estimators
            proba = self.predict_proba(y)
            for yi in range(y.shape[0]):
                # probabilities and corresponding indices in descending order
                idx_sort = np.argsort(proba[yi, :])[::-1]
                nr_proba = np.searchsorted(np.cumsum(proba[yi, idx_sort]), n_summands_or_proba) + 1
                for argproba in idx_sort[:nr_proba]:
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse(y[yi], A_eff, argproba, Cy_invs[argproba])
                h_est[yi, :] /= np.sum(proba[yi, idx_sort[:nr_proba]])
        return h_est


    def _prepare_for_prediction(self, y, A, snr_dB, n_bits, quantizer_type, quantizer):
        sigma2 = 10 ** (-snr_dB / 10)
        Cn = sigma2 * np.eye(A.shape[0])

        self._means = np.squeeze(A @ np.expand_dims(self.means, 2))
        self._covs = A @ self.covs @ A.conj().T
        for k in range(self.n_components):
            self._covs[k] += Cn
        # pre-compute Bussgang gain matrix for all components
        A_buss = np.zeros_like(self._covs)
        for i in range(A_buss.shape[0]):
            if n_bits == 1:
                A_buss[i] = np.sqrt(2 / np.pi) * np.diag(1 / np.sqrt(np.diag(self._covs[i])))
            else:
                if quantizer_type == 'uniform':
                    A_buss[i] = quant_uni.get_Bussgang_matrix(snr_dB=snr_dB, n_bits=n_bits, Cy=self._covs[i])
                elif quantizer_type == 'lloyd':
                    A_buss[i] = quant_lloyd.get_Bussgang_matrix(n_bits=n_bits, Cy=self._covs[i],
                                                                quantizer=quantizer)

        # update GMM means by Bussgang gain
        for comp in range(self._covs.shape[0]):
            self._means[comp] = np.squeeze(A_buss[comp] @ self._means[comp, :, None])

        # update covariance matrices
        if n_bits != np.inf:
            covs_gm_quant = np.zeros_like(self._covs)
            if n_bits == 1:
                for i in range(self._covs.shape[0]):
                    Psi_12 = np.real(np.diag(1 / np.sqrt(np.diag(self._covs[i, :, :]))))
                    inner_real = np.real(Psi_12 @ np.real(self._covs[i, :, :]) @ Psi_12)
                    inner_imag = np.real(Psi_12 @ np.imag(self._covs[i, :, :]) @ Psi_12)
                    inner_real[inner_real > 1] = 1.0
                    inner_imag[inner_imag > 1] = 1.0
                    inner_real[inner_real < -1] = -1.0
                    inner_imag[inner_imag < -1] = -1.0
                    covs_gm_quant[i, :, :] = 2 / np.pi * (np.arcsin(inner_real) + 1j * np.arcsin(inner_imag))
            else:
                for i in range(self._covs.shape[0]):
                    beta = np.clip(np.real(np.mean(np.diag(A_buss[i]))), 0, 1)
                    covs_gm_quant[i] = beta**2 * self._covs[i] + (1 - beta**2) * np.diag(np.diag(self._covs[i]))

            self._covs = covs_gm_quant.copy()
        self._inv_covs = np.zeros_like(self._covs)
        for i in range(self.n_components):
            self._inv_covs[i] = scipy.linalg.pinvh(self._covs[i])

        # compute effective observation matrix
        A_eff = A_buss @ A

        return self._inv_covs, A_eff


    def _lmmse(self, y, A, k, Cy_inv):
        return self.means[k] + self.covs[k] @ A[k].conj().T @ (Cy_inv @ (y - A[k] @ self.means[k]))


    def _initialize(self, data, maxiter=200, tol=1e-4):
        # Run K-means
        Kmeans = cluster.KMeans(n_clusters=self.n_components, n_init=1,
                                ).fit(ut.cplx2real(data, axis=1))
        self._means = ut.real2cplx(Kmeans.cluster_centers_, axis=1)
        if self.zero_mean:
            self._means[:] = 0.0
        #if init_ppca:
        #    self.kmeans_rs = Kmeans.labels_
        del Kmeans

        # Randomly assign factor loadings
        self._lambdas = (np.random.randn(self.n_components, self.D, self.M) +
                         1j * np.random.randn(self.n_components, self.D, self.M)) / np.sqrt(
            self.max_condition_number) / np.sqrt(2)

        # Set (high rank) variance to variance of all data, along a dimension
        self._psis = np.tile(np.var(data, axis=0)[None, :], (self.n_components, 1))

        # Set initial covs
        self._update_covs()

        # Randomly assign the amplitudes.
        self.amps = np.random.rand(self.n_components)
        self.amps /= np.sum(self.amps)


    def run_em(self, data):
        """
        Run the EM algorithm.
        """
        L = -np.inf
        for i in range(self.maxiter):
            self._EM_per_component(data, self.PPCA)
            newL = self.logLs.sum()
            self.L_all.append(newL)
            print(f'Iteration {i} | lower bound: {newL:.5f}', end='\r')
            dL = np.abs((newL - L) / newL)
            if i > 5 and dL < self.tol:
                break
            L = newL

        if i < self.maxiter - 1:
            if self.verbose:
                print("EM converged after {0} iterations".format(i))
                print("Final NLL = {0}".format(-newL))
        else:
            print("\nWarning: EM didn't converge after {0} iterations"
                  .format(i))


    def _EM_per_component(self, data, PPCA):
        # resposibilities and likelihoods
        self.logLs, self.rs = self._calc_probs(data)
        sumrs = np.sum(self.rs, axis=1)

        #pre-compute betas
        betas = np.transpose(self._lambdas.conj(), [0, 2, 1]) @ self._inv_covs

        for k in range(self.n_components):
            #E-step: Calculation of latents per component
            # latent values
            if self.zero_mean:
                latents = betas[k] @ data.T
            else:
                latents = betas[k] @ (data.T - self._means[k, :, None])

            # latent empirical covariance
            step1 = latents[:, None, :] * latents[None, :, :].conj()
            step2 = betas[k] @ self._lambdas[k]
            latent_covs = np.eye(self.M)[:, :, None] - step2[:, :, None] + step1

            #M-step: Calculation of new parameters per component
            lambdalatents = self._lambdas[k] @ latents
            if self.zero_mean:
                self._means[k] = 0.0
            else:
                self._means[k] = np.sum(self.rs[k] * (data.T - lambdalatents), axis=1) / sumrs[k]

            zeroed = data.T - self._means[k, :, None]
            self._lambdas[k] = np.dot(np.dot(zeroed[:, None, :] * latents[None, :, :].conj(), self.rs[k]),
                                      inv(np.dot(latent_covs, self.rs[k])))
            psis = np.real(np.dot((zeroed - lambdalatents) * zeroed.conj(), self.rs[k]) / sumrs[k])
            self._psis[k] = np.clip(psis, 1e-6, np.Inf)
            if PPCA:
                self._psis[k] = np.mean(self._psis[k]) * np.ones(self.D)
            self.amps[k] = sumrs[k] / data.shape[0]

        if self.lock_psis:
            psi = np.dot(sumrs, self._psis) / np.sum(sumrs)
            self._psis = np.full_like(self._psis, psi)
        self._update_covs()


    def _update_covs(self):
        """
        Update self.cov for responsibility, logL calc
        """
        self._covs = self._lambdas @ np.transpose(self._lambdas.conj(), [0,2,1])
        for k in range(self.n_components):
            self._covs[k] += np.diag(self._psis[k])
        self._inv_covs = self._invert_cov_all()

    def _calc_probs(self, data):
        """
        Calculate log likelihoods, responsibilites for each datum
        under each component.
        """
        logrs = np.zeros((self.n_components, self.N))
        #pre-compute logdets
        sgn, logdet = np.linalg.slogdet(self._covs)
        for k in range(self.n_components):
            logrs[k] = np.log(self.amps[k]) + self._log_multi_gauss_nodet(k, data) - logdet[k]

        # here lies some ghetto log-sum-exp...
        # nothing like a little bit of overflow to make your day better!
        L = self._log_sum(logrs)
        logrs -= L[None, :]
        sumrs = np.sum(np.exp(logrs), axis=1)
        logrs[sumrs < self.rs_clip, :] = np.log(self.rs_clip)
        return L, np.exp(logrs)


    def predict_proba(self, data):
        """
        Calculate responsibilites.
        """
        logrs = np.zeros((self.n_components, data.shape[0]))
        for k in range(self.n_components):
            logrs[k] = np.log(self.amps[k]) + self._log_multi_gauss(k, data)

        # here lies some ghetto log-sum-exp...
        # nothing like a little bit of overflow to make your day better!
        L = self._log_sum(logrs)
        logrs -= L[None, :]
        # if self.rs_clip > 0.0:
        #    logrs = np.clip(logrs, np.log(self.rs_clip), np.Inf)
        return np.exp(logrs).T


    def predict_proba_max(self, data):
        """
        Calculate label with highest responsibility (argmax).
        """
        logrs = np.zeros((self.n_components, data.shape[0]))
        for k in range(self.n_components):
            logrs[k] = np.log(self.amps[k]) + self._log_multi_gauss(k, data)
        return np.exp(logrs).argmax(axis=0)
        #return logrs.argmax(axis=0)


    def _log_multi_gauss(self, k, data):
        """
        Gaussian log likelihood of the data for component k.
        """
        sgn, logdet = np.linalg.slogdet(self._covs[k])
        #if not (sgn > 0):
        #    print(f'logdet: {logdet}, sgn: {sgn}')
        #assert sgn > 0
        X1 = (data - self._means[k]).T
        X2 = self._inv_covs[k] @ X1
        p = np.sum(X1.conj() * X2, axis=0)
        return np.real(- np.log(np.pi) * data.shape[1] - logdet - p)


    def _log_multi_gauss_nodet(self, k, data):
        """
        Gaussian log likelihood of the data for component k.
        """
        X1 = (data - self._means[k]).T
        X2 = self._inv_covs[k] @ X1
        p = np.sum(X1.conj() * X2, axis=0)
        return np.real(- np.log(np.pi) * data.shape[1] - p)


    def _log_sum(self, loglikes):
        """
        Calculate sum of log likelihoods
        """
        loglikes = np.atleast_2d(loglikes)
        a = np.max(loglikes, axis=0)
        return a + np.log(np.sum(np.exp(loglikes - a[None, :]), axis=0))


    def _invert_cov(self, k):
        """
        Calculate inverse covariance of mofa or ppca model,
        using inversion lemma
        """
        psiI = 1 / self._psis[k]
        inv_inner = inv(np.eye(self.M) + self._lambdas[k].conj().T @ np.diag(psiI) @ self._lambdas[k])
        return np.diag(psiI) - psiI[:, None] * (self._lambdas[k] @ inv_inner @ self._lambdas[k].conj().T) * psiI

    def _invert_cov_all(self):
        """
        Calculate inverse covariance of mofa or ppca model,
        using inversion lemma of all components at once.
        """
        psiI = 1 / self._psis
        inv_inner = np.linalg.pinv(np.eye(self.M)[None, :, :] + (np.transpose(self._lambdas.conj(), [0, 2, 1]) * psiI[:, None, :]) @ self._lambdas)
        step = psiI[:, :, None] * (self._lambdas @ inv_inner @ np.transpose(self._lambdas.conj(), [0,2,1])) * psiI[:, None, :]
        for k in range(self.n_components):
            step[k] -= np.diag(psiI[k])
        return -step