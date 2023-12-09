import numpy as np
import scipy.stats
import modules.utils as ut
from scipy import linalg as scilinalg
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn import cluster
from modules import uniform_quantizer as quant_uni
from modules import lloyd_max_quantizer as quant_lloyd



def compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.

    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.

    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type == 'full':
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features), dtype=complex)
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = scilinalg.cholesky(covariance, lower=True)
            except scilinalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = scilinalg.solve_triangular(cov_chol, np.eye(n_features), lower=True).T.conj()
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances).conj()
    return precisions_chol


def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.
    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    n_features : int
        Number of features.
    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape
        log_det_chol = np.sum(
            np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )
    elif covariance_type == "diag":
        log_det_chol = np.sum(np.log(matrix_chol), axis=1)
    else:
        log_det_chol = n_features * (np.log(matrix_chol))
    return log_det_chol


class Gmm_nbit:
    def __init__(self, *gmm_args, **gmm_kwargs):
        self.gm = GaussianMixture(*gmm_args, **gmm_kwargs)
        self.means_cplx = None
        self.covs_cplx = None
        self.fft_covs = None
        self.fft_means = None
        self.chol = None
        self.params = dict()
        self.F2 = None

    def fit(self, h, blocks=None, zero_mean=False):
        if zero_mean:
            self.params['zero_mean'] = True
        else:
            self.params['zero_mean'] = False
        """
        Fit an sklearn Gaussian mixture model using complex data h.
        """
        if self.gm.covariance_type == 'circulant':
            self.gm.covariance_type = 'diag'
            dft_matrix = np.fft.fft(np.eye(h.shape[-1], dtype=complex)) / np.sqrt(h.shape[-1])
            self.fit_cplx(np.fft.fft(h, axis=1) / np.sqrt(h.shape[-1]))
            self.fft_covs = self.gm.covariances_
            self.fft_means = self.gm.means_
            self.means_cplx = self.gm.means_ @ dft_matrix.conj()
            self.covs_cplx = np.zeros([self.means_cplx.shape[0], self.means_cplx.shape[-1],
                                       self.means_cplx.shape[-1]], dtype=complex)
            for i in range(self.means_cplx.shape[0]):
                self.covs_cplx[i] = dft_matrix.conj().T @ np.diag(self.gm.covariances_[i]) @ dft_matrix
            self.chol = compute_precision_cholesky(self.covs_cplx, 'full')
            self.gm.covariance_type = 'full'
            self.gm.means_ = self.means_cplx
            self.gm.precisions_cholesky_ = self.chol
            self.gm.covariances_ = self.covs_cplx
        elif self.gm.covariance_type == 'block-circulant':
            self.gm.covariance_type = 'diag'
            n_1, n_2 = blocks
            F1 = np.fft.fft(np.eye(n_1)) / np.sqrt(n_1)
            F2 = np.fft.fft(np.eye(n_2)) / np.sqrt(n_2)
            dft_matrix = np.kron(F1, F2)
            self.F2 = dft_matrix
            self.fit_cplx(np.squeeze(dft_matrix @ np.expand_dims(h, 2)))
            self.means_cplx = self.gm.means_ @ dft_matrix.conj()
            self.covs_cplx = np.zeros([self.means_cplx.shape[0], self.means_cplx.shape[-1],
                                       self.means_cplx.shape[-1]], dtype=complex)
            for i in range(self.means_cplx.shape[0]):
                self.covs_cplx[i] = dft_matrix.conj().T @ np.diag(self.gm.covariances_[i]) @ dft_matrix
            self.chol = compute_precision_cholesky(self.covs_cplx, 'full')
            self.gm.covariance_type = 'full'
            self.gm.means_ = self.means_cplx
            self.gm.precisions_cholesky_ = self.chol
        elif self.gm.covariance_type == 'full':
            self.fit_cplx(h)
            self.means_cplx = self.gm.means_.copy()
            self.covs_cplx = self.gm.covariances_.copy()
            self.chol = self.gm.precisions_cholesky_.copy()
        elif self.gm.covariance_type == 'toeplitz':
            self.params['inv-em'] = True
            self.gm.covariance_type = 'full'
            n_1 = h.shape[1]
            self.F2 = np.fft.fft(np.eye(2 * n_1))[:, :n_1] / np.sqrt(2 * n_1)
            self.fit_cplx(h)
            self.means_cplx = self.gm.means_.copy()
            self.covs_cplx = self.gm.covariances_.copy()
            self.chol = self.gm.precisions_cholesky_.copy()
        elif self.gm.covariance_type == 'block-toeplitz':
            self.params['inv-em'] = True
            self.gm.covariance_type = 'full'
            n_1, n_2 = blocks
            F2_1 = np.fft.fft(np.eye(2 * n_1))[:, :n_1] / np.sqrt(2 * n_1)
            F2_2 = np.fft.fft(np.eye(2 * n_2))[:, :n_2] / np.sqrt(2 * n_2)
            self.F2 = np.kron(F2_1, F2_2)
            self.fit_cplx(h)
            self.means_cplx = self.gm.means_.copy()
            self.covs_cplx = self.gm.covariances_.copy()
            self.chol = self.gm.precisions_cholesky_.copy()
        else:
            raise NotImplementedError(f'Fitting for covariance_type = {self.gm.covariance_type} is not implemented.')


    def estimate_from_y(self, y, snr_dB, n_antennas, A=None, n_summands_or_proba=1, n_bits=1,
                        quantizer_type='uniform', quantizer=None):
        """
        Use the noise covariance matrix and the matrix A to update the
        covariance matrices of the Gaussian mixture model. This GMM is then
        used for channel estimation from y.

        Args:
            y: A 2D complex numpy array.
            snr_dB: The SNR in dB.
            n_antennas: The dimension of the channels.
            A: A complex observation matrix.
            n_summands_or_proba:
                If equal to 'all', compute the sum of all LMMSE estimates.
                If equal to an integer, compute the sum of the top (highest
                    component probabilities) n_summands_or_proba LMMSE
                    estimates.
                If equal to a float, compute the sum of as many LMMSE estimates
                    as are necessary to reach at least a cumulative component
                    probability of n_summands_or_proba.
            n_bits: Number of quantization bits.
            quantizer_type: The quantizer type (uniform, lloyd).
            quantizer: The quantizer dictionary for each SNR.
        """

        if A is None:
            A = np.eye(n_antennas, dtype=y.dtype)
        y_for_prediction, covs_Cr_inv, covs_Cy, A_eff = self._prepare_for_prediction(y, A, snr_dB, n_bits,
                                                                                     quantizer_type, quantizer)

        h_est = np.zeros([y.shape[0], A.shape[-1]], dtype=complex)
        if isinstance(n_summands_or_proba, int):
            # n_summands_or_proba represents a number of summands

            if n_summands_or_proba == 1:
                # use predicted label to choose the channel covariance matrix
                labels = self._predict_cplx(y_for_prediction)
                for yi in range(y.shape[0]):
                    mean_h = self.means_cplx[labels[yi], :]
                    h_est[yi] = self._lmmse_formula(y[yi], mean_h,
                                    self.covs_cplx[labels[yi]] @ A_eff[labels[yi]].conj().T, covs_Cr_inv[labels[yi]],
                                    A_eff[labels[yi]] @ mean_h)
            else:
                # use predicted probabilites to compute weighted sum of estimators
                proba = self.predict_proba_cplx(y_for_prediction)
                for yi in range(y.shape[0]):
                    # indices for probabilites in descending order
                    idx_sort = np.argsort(proba[yi, :])[::-1]
                    for argproba in idx_sort[:n_summands_or_proba]:
                        mean_h = self.means_cplx[argproba, :]
                        h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(y[yi, :], mean_h,
                                              self.covs_cplx[argproba, :, :] @ A_eff[argproba].conj().T,
                                              covs_Cr_inv[argproba, :, :], A_eff[argproba] @ mean_h)
                    h_est[yi, :] /= np.sum(proba[yi, idx_sort[:n_summands_or_proba]])
        elif n_summands_or_proba == 'all':
            # use all predicted probabilities to compute weighted sum of estimators
            proba = self.predict_proba_cplx(y_for_prediction)
            for yi in range(y.shape[0]):
                for argproba in range(proba.shape[1]):
                    mean_h = self.means_cplx[argproba, :]
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(y[yi, :], mean_h,
                                      self.covs_cplx[argproba, :, :] @ A_eff[argproba].conj().T,
                                      covs_Cr_inv[argproba, :, :], A_eff[argproba] @ mean_h)
        else:
            # n_summands_or_proba represents a probability
            # use predicted probabilites to compute weighted sum of estimators
            proba = self.predict_proba_cplx(y_for_prediction)
            for yi in range(y.shape[0]):
                # probabilities and corresponding indices in descending order
                idx_sort = np.argsort(proba[yi, :])[::-1]
                nr_proba = np.searchsorted(np.cumsum(proba[yi, idx_sort]), n_summands_or_proba) + 1
                for argproba in idx_sort[:nr_proba]:
                    mean_h = self.means_cplx[argproba, :]
                    h_est[yi, :] += proba[yi, argproba] * self._lmmse_formula(y[yi, :], mean_h,
                                      self.covs_cplx[argproba, :, :] @ A_eff[argproba].conj().T,
                                      covs_Cr_inv[argproba, :, :], A_eff[argproba] @ mean_h)
                h_est[yi, :] /= np.sum(proba[yi, idx_sort[:nr_proba]])
        return h_est


    def _prepare_for_prediction(self, y, A, snr_dB, n_bits=1, quantizer_type='lloyd', quantizer=None):
        """
        Replace the GMM's means and covariance matrices by the means and
        covariance matrices of the observation. Further, in case of diagonal
        matrices, FFT-transform the observation.
        """
        sigma2 = 10 ** (-snr_dB / 10)

        if self.gm.covariance_type == 'full':
            # update GMM means
            Am = np.squeeze(np.matmul(A, np.expand_dims(self.means_cplx, axis=2)))
            if len(Am.shape) == 1:
                Am = np.expand_dims(Am, 0)

            # handle the case of only one GMM component
            if Am.ndim == 1:
                self.gm.means_ = Am[None, :]
            else:
                self.gm.means_ = Am

            # compute cov of unquantized observation
            covs_gm = self.covs_cplx.copy()
            covs_gm = np.matmul(np.matmul(A, covs_gm), A.conj().T)
            sigma2_diag = sigma2 * np.eye(covs_gm.shape[-1])
            for i in range(covs_gm.shape[0]):
                covs_gm[i, :, :] = covs_gm[i, :, :] + sigma2_diag

            #pre-compute Bussgang gain matrix for all components
            A_buss = np.zeros_like(covs_gm)
            for i in range(A_buss.shape[0]):
                if n_bits == 1:
                    A_buss[i] = np.sqrt(2 / np.pi) * np.diag(1 / np.sqrt(np.diag(covs_gm[i])))
                elif n_bits == np.inf:
                    A_buss[i] = np.eye(covs_gm.shape[-1])
                else:
                    if quantizer_type == 'uniform':
                        A_buss[i] = quant_uni.get_Bussgang_matrix(snr_dB=snr_dB, n_bits=n_bits, Cy=covs_gm[i])
                    elif quantizer_type == 'lloyd':
                        A_buss[i] = quant_lloyd.get_Bussgang_matrix(n_bits=n_bits, Cy=covs_gm[i], quantizer=quantizer)

            # update GMM means by Bussgang gain
            for comp in range(covs_gm.shape[0]):
                self.gm.means_[comp] = A_buss[comp] @ Am[comp]

            # update covariance matrices
            covs_gm_quant = np.zeros_like(covs_gm)
            if n_bits == 1:
                for i in range(self.gm.n_components):
                    Psi_12 = np.real(np.diag(1 / np.sqrt(np.diag(covs_gm[i, :, :]))))
                    inner_real = np.real(Psi_12 @ np.real(covs_gm[i, :, :]) @ Psi_12)
                    inner_imag = np.real(Psi_12 @ np.imag(covs_gm[i, :, :]) @ Psi_12)
                    inner_real[inner_real > 1] = 1.0
                    inner_imag[inner_imag > 1] = 1.0
                    inner_real[inner_real < -1] = -1.0
                    inner_imag[inner_imag < -1] = -1.0
                    covs_gm_quant[i, :, :] = 2 / np.pi * (np.arcsin(inner_real) + 1j * np.arcsin(inner_imag))
            elif n_bits == np.inf:
                covs_gm_quant = covs_gm.copy()
            else:
                for i in range(self.gm.n_components):
                    beta = np.clip(np.real(np.mean(np.diag(A_buss[i]))), 0, 1)
                    covs_gm_quant[i] = beta**2 * covs_gm[i] + (1 - beta**2) * np.diag(np.diag(covs_gm[i])) #np.eye(covs_gm.shape[-1])

            self.gm.covariances_ = covs_gm_quant.copy()
            self.gm.precisions_cholesky_ = compute_precision_cholesky(covs_gm_quant, covariance_type='full')

            # update GMM feature number
            self.gm.n_features_in_ = A.shape[0]

            y_for_prediction = y
        else:
            raise NotImplementedError(f'Estimation for covariance_type = {self.gm.covariance_type} is not implemented.')

        # precompute the inverse matrices
        #cov_noise = sigma2 * np.eye(y.shape[1], dtype=complex)
        covs_gm_quan_inv = np.zeros_like(covs_gm_quant)
        for i in range(self.covs_cplx.shape[0]):
            covs_gm_quan_inv[i, :, :] = np.linalg.pinv(covs_gm_quant[i, :, :])

        #compute effective observation matrix
        A_eff = A_buss @ A

        return y_for_prediction, covs_gm_quan_inv, covs_gm, A_eff


    def _lmmse_formula(self, y, mean_h, cov_h, cov_y_inv, mean_y):
        return mean_h + cov_h @ (cov_y_inv @ (y - mean_y))


    def _predict_cplx(self, X):
        """Predict the labels for the data samples in X using trained model.

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    List of n_features-dimensional data points. Each row
                    corresponds to a single data point.

                Returns
                -------
                labels : array, shape (n_samples,)
                    Component labels.
                """
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def predict_proba_cplx(self, X):
        """Predict posterior probability of each component given the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        resp : array, shape (n_samples, n_components)
            Returns the probability each Gaussian (state) in
            the model given each sample.
        """
        _, log_resp = self._estimate_log_prob_resp(X)
        return np.exp(log_resp)

    def _estimate_weighted_log_prob(self, X):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return self._estimate_log_prob(X) + self._estimate_log_weights()

    def _estimate_log_weights(self):
        return np.log(self.gm.weights_)

    def _estimate_log_prob(self, X):
        return self._estimate_log_gaussian_prob(X, self.gm.means_, self.gm.precisions_cholesky_,  self.gm.covariance_type)

    def _estimate_log_gaussian_prob(self, X, means, precisions_chol, covariance_type):
        """Estimate the log Gaussian probability.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        means : array-like of shape (n_components, n_features)
        precisions_chol : array-like
            Cholesky decompositions of the precision matrices.
            'full' : shape of (n_components, n_features, n_features)
            'tied' : shape of (n_features, n_features)
            'diag' : shape of (n_components, n_features)
            'spherical' : shape of (n_components,)
        covariance_type : {'full', 'tied', 'diag', 'spherical'}
        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape
        n_components, _ = means.shape
        # The determinant of the precision matrix from the Cholesky decomposition
        # corresponds to the negative half of the determinant of the full precision
        # matrix.
        # In short: det(precision_chol) = - det(precision) / 2
        log_det = np.real(_compute_log_det_cholesky(precisions_chol, covariance_type, n_features))

        if covariance_type == "full":
            log_prob = np.empty((n_samples, n_components))
            for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
                y = np.dot(X, prec_chol.conj()) - np.dot(mu, prec_chol.conj())
                log_prob[:, k] = np.sum(np.abs(y)**2, axis=1)

        elif covariance_type == "diag":
            precisions = np.abs(precisions_chol) ** 2
            log_prob = (
                    np.sum((np.abs(means) ** 2 * precisions), 1)
                    - 2.0 * np.real(np.dot(X, (means.conj() * precisions).T))
                    + np.dot(np.abs(X) ** 2, precisions.T)
            )
        elif covariance_type == "spherical":
            precisions = np.abs(precisions_chol) ** 2
            log_prob = (
                    np.sum(np.abs(means) ** 2, 1) * precisions
                    - 2 * np.real(np.dot(X, means.conj().T) * precisions)
                    + np.outer((X.conj() * X).sum(axis=1), precisions)
            )
        # Since we are using the precision of the Cholesky decomposition,
        # `- log_det_precision` becomes `+ 2 * log_det_precision_chol`
        return -(n_features * np.log(np.pi) + log_prob) + 2*log_det

    def fit_cplx(self, X, y=None):
        """Estimate model parameters with the EM algorithm.

        The method fits the model ``n_init`` times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for ``max_iter``
        times until the change of likelihood or lower bound is less than
        ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
        If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
        initialization is performed upon the first call. Upon consecutive
        calls, training starts where it left off.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        self.fit_predict(X, y)
        return self

    def fit_predict(self, X, y=None):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        #X = _check_X(X, self.n_components, ensure_min_samples=2)
        self.gm._check_n_features(X, reset=True)
        self.gm._check_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not(self.gm.warm_start and hasattr(self, 'converged_'))
        n_init = self.gm.n_init if do_init else 1

        max_lower_bound = -np.infty
        self.gm.converged_ = False

        random_state = ut.check_random_state(self.gm.random_state)

        n_samples, _ = X.shape
        for init in range(n_init):
            self.gm._print_verbose_msg_init_beg(init)

            if do_init:
                self._initialize_parameters(X, random_state)

            lower_bound = (-np.infty if do_init else self.gm.lower_bound_)

            for n_iter in range(1, self.gm.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp = self._e_step(X)
                self._m_step(X, log_resp)
                lower_bound = self.gm._compute_lower_bound(
                    log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                self.gm._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < self.gm.tol:
                    self.gm.converged_ = True
                    break

            self.gm._print_verbose_msg_init_end(lower_bound)

            if lower_bound > max_lower_bound:
                max_lower_bound = lower_bound
                best_params = self.gm._get_parameters()
                best_n_iter = n_iter

        if not self.gm.converged_:
            warnings.warn('Initialization %d did not converge. '
                          'Try different init parameters, '
                          'or increase max_iter, tol '
                          'or check for degenerate data.'
                          % (init + 1), ConvergenceWarning)

        self._set_parameters(best_params)
        self.gm.n_iter_ = best_n_iter
        self.gm.lower_bound_ = max_lower_bound

        # Always do a final e-step to guarantee that the labels returned by
        # fit_predict(X) are always consistent with fit(X).predict(X)
        # for any value of max_iter and tol (and any random_state).
        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)


    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        n_samples, _ = X.shape

        if self.gm.init_params == 'kmeans':
            resp = np.zeros((n_samples, self.gm.n_components))
            X_real = ut.cplx2real(X, axis=1)
            label = cluster.KMeans(n_clusters=self.gm.n_components, n_init=1,
                                   random_state=random_state).fit(X_real).labels_
            resp[np.arange(n_samples), label] = 1
        elif self.gm.init_params == 'random':
            resp = random_state.rand(n_samples, self.gm.n_components)
            resp /= resp.sum(axis=1)[:, np.newaxis]
        else:
            raise ValueError("Unimplemented initialization method '%s'"
                             % self.gm.init_params)
        self._initialize(X, resp)

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        resp : array-like of shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = self.estimate_gaussian_parameters(
            X, resp, self.gm.reg_covar, self.gm.covariance_type)
        weights /= n_samples

        self.gm.weights_ = (weights if self.gm.weights_init is None
                         else self.gm.weights_init)
        self.gm.means_ = means if self.gm.means_init is None else self.gm.means_init

        if self.gm.precisions_init is None:
            self.gm.covariances_ = covariances
            self.gm.precisions_cholesky_ = compute_precision_cholesky(
                covariances, self.gm.covariance_type)
            if 'inv-em' in self.params:
                self.gm.Sigma = np.zeros([self.gm.n_components, self.F2.shape[0]])
                for k in range(self.gm.n_components):
                    self.gm.Sigma[k] = np.real(np.diag(self.F2 @ covariances[k] @ self.F2.conj().T))
                    self.gm.Sigma[k][self.gm.Sigma[k] < self.gm.reg_covar] = self.gm.reg_covar
        elif self.gm.covariance_type == 'full':
            self.gm.precisions_cholesky_ = np.array(
                [scipy.linalg.cholesky(prec_init, lower=True)
                 for prec_init in self.gm.precisions_init])
        else:
            self.gm.precisions_cholesky_ = np.sqrt(self.gm.precisions_init)


    def _e_step(self, X):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp


    def _estimate_log_prob_resp(self, X):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp


    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        if 'inv-em' in self.params:
            self.gm.weights_, self.gm.means_, self.gm.covariances_ = (
                self.estimate_gaussian_parameters(X, np.exp(log_resp), self.gm.reg_covar,
                                                'inv-em'))
        else:
            self.gm.weights_, self.gm.means_, self.gm.covariances_ = (
                self.estimate_gaussian_parameters(X, np.exp(log_resp), self.gm.reg_covar,
                                              self.gm.covariance_type))
        self.gm.weights_ /= n_samples
        self.gm.precisions_cholesky_ = compute_precision_cholesky(
            self.gm.covariances_, self.gm.covariance_type)


    def _set_parameters(self, params):
        (self.gm.weights_, self.gm.means_, self.gm.covariances_,
         self.gm.precisions_cholesky_) = params

        # Attributes computation
        _, n_features = self.gm.means_.shape

        if self.gm.covariance_type == 'full':
            self.gm.precisions_ = np.empty(self.gm.precisions_cholesky_.shape, dtype=complex)
            for k, prec_chol in enumerate(self.gm.precisions_cholesky_):
                self.gm.precisions_[k] = np.dot(prec_chol, prec_chol.T.conj())
        else:
            self.gm.precisions_ = np.abs(self.gm.precisions_cholesky_) ** 2


    def estimate_gaussian_parameters(self, X, resp, reg_covar, covariance_type):
        """Estimate the Gaussian distribution parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data array.

        resp : array-like of shape (n_samples, n_components)
            The responsibilities for each data sample in X.

        reg_covar : float
            The regularization added to the diagonal of the covariance matrices.

        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.

        Returns
        -------
        nk : array-like of shape (n_components,)
            The numbers of data samples in the current components.

        means : array-like of shape (n_components, n_features)
            The centers of the current components.

        covariances : array-like
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        """
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X) / nk[:, np.newaxis]
        if self.params['zero_mean']:
            means = np.zeros_like(means)
        covariances = {"full": self.estimate_gaussian_covariances_full,
                       "diag": self.estimate_gaussian_covariances_diag,
                       "inv-em": self.estimate_gaussian_covariances_inv,
                       "spherical": self.estimate_gaussian_covariances_spherical,
                       }[covariance_type](resp, X, nk, means, reg_covar)
        return nk, means, covariances

    def estimate_gaussian_covariances_full(self, resp, X, nk, means, reg_covar):
        """Estimate the full covariance matrices.

        Parameters
        ----------
        resp : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, n_features, n_features)
            The covariance matrix of the current components.
        """
        n_components, n_features = means.shape
        covariances = np.empty((n_components, n_features, n_features), dtype=complex)
        for k in range(n_components):
            diff = X - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff.conj()) / nk[k]
            covariances[k].flat[::n_features + 1] += reg_covar
        return covariances

    def estimate_gaussian_covariances_diag(self, resp, X, nk, means, reg_covar):
        """Estimate the diagonal covariance vectors.

        Parameters
        ----------
        responsibilities : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, n_features)
            The covariance vector of the current components.
        """
        avg_X2 = np.dot(resp.T, X * X.conj()) / nk[:, np.newaxis]
        avg_means2 = np.abs(means) ** 2
        avg_X_means = means.conj() * np.dot(resp.T, X) / nk[:, np.newaxis]
        return avg_X2 - 2.0 * np.real(avg_X_means) + avg_means2 + reg_covar

    def estimate_gaussian_covariances_inv(self, resp, X, nk, means, reg_covar):
        """Estimate the Topelitz-structured covariance matrices.
        Method is used from T. A. Barton and D. R. Fuhrmann, "Covariance estimation for multidimensional data
        using the EM algorithm," Proceedings of 27th Asilomar Conference on Signals, Systems and Computers, 1993,
        pp. 203-207 vol.1.

        Parameters
        ----------
        resp : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, n_features, n_features)
            The covariance matrix of the current components.
        """
        n_components, n_features = means.shape
        covariances = np.empty((n_components, n_features, n_features), dtype=complex)
        Cinv = np.linalg.pinv(self.gm.covariances_, hermitian=True)
        for k in range(n_components):
            diff = X - means[k]
            covariances[k] = np.dot(resp[:, k] * diff.T, diff.conj()) / nk[k]
            Theta = np.real(self.F2 @ (Cinv[k] @ covariances[k] @ Cinv[k] - Cinv[k]) @ self.F2.conj().T)
            self.gm.Sigma[k] = self.gm.Sigma[k] + np.diag(np.multiply(np.multiply(self.gm.Sigma[k], Theta), self.gm.Sigma[k]))
            self.gm.Sigma[k][self.gm.Sigma[k] < reg_covar] = reg_covar
            covariances[k] = np.multiply(self.F2.conj().T, self.gm.Sigma[k]) @ self.F2
            covariances[k].flat[::n_features + 1] += reg_covar
        return covariances

    def estimate_gaussian_covariances_spherical(self, resp, X, nk, means, reg_covar):
        """Estimate the full covariance matrices.

        Parameters
        ----------
        resp : array-like of shape (n_samples, n_components)

        X : array-like of shape (n_samples, n_features)

        nk : array-like of shape (n_components,)

        means : array-like of shape (n_components, n_features)

        reg_covar : float

        Returns
        -------
        covariances : array, shape (n_components, n_features, n_features)
            The covariance matrix of the current components.
        """
        return self.estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar).mean(1)