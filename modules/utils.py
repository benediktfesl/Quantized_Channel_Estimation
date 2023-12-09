import numpy as np
import torch
from scipy import linalg
import os
from typing import Tuple
import multiprocessing as mp
import csv
from modules.lloyd_max_quantizer import load_quantizer
from modules.uniform_quantizer import get_uniform_quant_step



def crandn(*arg, rng=np.random.default_rng()):
    return np.sqrt(0.5) * (rng.standard_normal(arg) + 1j * rng.standard_normal(arg))


class DummyArray:
    """Dummy object that just exists to hang __array_interface__ dictionaries
    and possibly keep alive a reference to a base array.
    """

    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base

def _maybe_view_as_subclass(original_array, new_array):
    if type(original_array) is not type(new_array):
        # if input was an ndarray subclass and subclasses were OK,
        # then view the result as that subclass.
        new_array = new_array.view(type=type(original_array))
        # Since we have done something akin to a view from original_array, we
        # should let the subclass finalize (if it has it implemented, i.e., is
        # not None).
        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array


def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
    """
    Create a view into the array with the given shape and strides.

    .. warning:: This function has to be used with extreme care, see notes.

    Parameters
    ----------
    x : ndarray
        Array to create a new.
    shape : sequence of int, optional
        The shape of the new array. Defaults to ``x.shape``.
    strides : sequence of int, optional
        The strides of the new array. Defaults to ``x.strides``.
    subok : bool, optional
        .. versionadded:: 1.10

        If True, subclasses are preserved.
    writeable : bool, optional
        .. versionadded:: 1.12

        If set to False, the returned array will always be readonly.
        Otherwise it will be writable if the original array was. It
        is advisable to set this to False if possible (see Notes).

    Returns
    -------
    view : ndarray

    See also
    --------
    broadcast_to: broadcast an array to a given shape.
    reshape : reshape an array.

    Notes
    -----
    ``as_strided`` creates a view into the array given the exact strides
    and shape. This means it manipulates the internal data structure of
    ndarray and, if done incorrectly, the array elements can point to
    invalid memory and can corrupt results or crash your program.
    It is advisable to always use the original ``x.strides`` when
    calculating new strides to avoid reliance on a contiguous memory
    layout.

    Furthermore, arrays created with this function often contain self
    overlapping memory, so that two elements are identical.
    Vectorized write operations on such arrays will typically be
    unpredictable. They may even give different results for small, large,
    or transposed arrays.
    Since writing to these arrays has to be tested and done with great
    care, you may want to use ``writeable=False`` to avoid accidental write
    operations.

    For these reasons it is advisable to avoid ``as_strided`` when
    possible.
    """
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)
    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)

    array = np.asarray(DummyArray(interface, base=x))
    # The route via `__interface__` does not preserve structured
    # dtypes. Since dtype should remain unchanged, we set it explicitly.
    array.dtype = x.dtype

    view = _maybe_view_as_subclass(x, array)

    if view.flags.writeable and not writeable:
        view.flags.writeable = False

    return view

def toeplitz(c, r=None):
    """
    Construct a Toeplitz matrix.

    The Toeplitz matrix has constant diagonals, with c as its first column
    and r as its first row. If r is not given, ``r == conjugate(c)`` is
    assumed.

    Parameters
    ----------
    c : array_like
        First column of the matrix.  Whatever the actual shape of `c`, it
        will be converted to a 1-D array.
    r : array_like, optional
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        ``[c[0], r[1:]]``.  Whatever the actual shape of `r`, it will be
        converted to a 1-D array.

    Returns
    -------
    A : (len(c), len(r)) ndarray
        The Toeplitz matrix. Dtype is the same as ``(c[0] + r[0]).dtype``.

    See Also
    --------
    circulant : circulant matrix
    hankel : Hankel matrix
    solve_toeplitz : Solve a Toeplitz system.

    Notes
    -----
    The behavior when `c` or `r` is a scalar, or when `c` is complex and
    `r` is None, was changed in version 0.8.0. The behavior in previous
    versions was undocumented and is no longer supported.

    Examples
    --------
    >>> from scipy.linalg import toeplitz
    >>> toeplitz([1,2,3], [1,4,5,6])
    array([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])
    >>> toeplitz([1.0, 2+3j, 4-1j])
    array([[ 1.+0.j,  2.-3.j,  4.+1.j],
           [ 2.+3.j,  1.+0.j,  2.-3.j],
           [ 4.-1.j,  2.+3.j,  1.+0.j]])

    """
    c = np.asarray(c).ravel()
    if r is None:
        r = c.conjugate()
    else:
        r = np.asarray(r).ravel()
    # Form a 1-D array containing a reversed c followed by r[1:] that could be
    # strided to give us toeplitz matrix.
    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = len(c), len(r)
    n = vals.strides[0]
    return as_strided(vals[len(c)-1:], shape=out_shp, strides=(-n, n)).copy()


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def mse_loss_cplx(output, target):
    loss = torch.mean(torch.abs(output - target)**2)
    return loss


def quant(inp, n_bits=1, thresholds=None, quant_labels=None):
    if n_bits == 1:
        return 1 / np.sqrt(2) * (np.sign(np.real(inp)) + 1j*np.sign(np.imag(inp)))
    else:
        val_r = np.real(inp.flatten())
        val_i = np.imag(inp.flatten())
        quant_index_r = list(np.digitize(val_r, thresholds))
        quant_index_i = list(np.digitize(val_i, thresholds))

        val_r = quant_labels[quant_index_r]
        val_i = quant_labels[quant_index_i]
        quant_val = val_r + 1j * val_i
        quant_val = np.resize(quant_val, inp.shape)

        return quant_val


def quant_torch(inp, n_bits=1, thresholds=None, quant_labels=None, device='cpu'):
    if n_bits == 1:
        return 1 / np.sqrt(2) * (torch.sign(torch.real(inp)) + 1j * torch.sign(torch.imag(inp)))
    else:
        val_r = torch.real(inp.flatten())
        val_i = torch.imag(inp.flatten())

        quant_index_r = np.digitize(val_r.cpu().numpy(), thresholds)
        quant_index_i = np.digitize(val_i.cpu().numpy(), thresholds)

        val_r = torch.as_tensor(quant_labels[quant_index_r], device=device, dtype=inp.dtype)
        val_i = torch.as_tensor(quant_labels[quant_index_i], device=device, dtype=inp.dtype)
        quant_val = val_r + 1j * val_i
        quant_val = quant_val.view(inp.shape)

        return quant_val


def quant_torch_fast(inp, n_bits=1, thresholds=None, quant_labels=None, device='cpu'):
    #thresholds = torch.tensor(thresholds, device=device, dtype=torch.float)
    if n_bits == 1:
        return 1 / torch.sqrt(torch.tensor(2.0, device=device)) * (torch.sign(torch.real(inp)) + 1j * torch.sign(torch.imag(inp)))
    else:
        val_r = torch.real(inp.flatten())
        val_i = torch.imag(inp.flatten())

        quant_index_r = torch.bucketize(val_r, thresholds).clamp(0, len(quant_labels) - 1)
        quant_index_i = torch.bucketize(val_i, thresholds).clamp(0, len(quant_labels) - 1)

        val_r = quant_labels[quant_index_r]
        val_i = quant_labels[quant_index_i]
        quant_val = val_r + 1j * val_i
        #quant_val = quant_val.view(inp.shape)
        return quant_val

def get_observation_nbit(h, snr, A=None, n_bits=1, thresholds=None, cluster=None, agc=False):
    if A is None:
        A = np.eye(h.shape[-1])
    y = np.squeeze(np.matmul(A, np.expand_dims(h, 2)))
    if h.shape[1] == 1:
        y = np.expand_dims(y, 1)
    y += 10 ** (-snr / 20) * crandn(*y.shape)
    if n_bits == 'inf' or n_bits == np.inf:
        return y
    else:
        return quant(y, n_bits, thresholds, cluster)


def get_observation_nbit_randSNR(h, snrs, A=None, n_bits=1, quantizer=None, snr_scaling=None):
    if A is None:
        A = np.eye(h.shape[-1])
    y = np.squeeze(np.matmul(A, np.expand_dims(h, 2)))
    snr_list = np.zeros([h.shape[0]])
    for i in range(h.shape[0]):
        if snr_scaling is not None:
            snr = np.random.choice(snrs, p=snr_scaling)
        else:
            snr = np.random.choice(snrs)
        snr_list[i] = snr
        y[i] += 10 ** (-snr / 20) * crandn(*y[i].shape)
        if n_bits != np.inf:
            y[i] = quant(y[i], n_bits, quantizer[snr][0], quantizer[snr][1])
    return y, snr_list


def get_observation_nbit_randSNR_torch(h, snrs, A=None, n_bits=1, quantizer=None, snr_scaling=None, device='cpu'):
    if A is None:
        #A = torch.eye(h.shape[-1], dtype=h.type, device=device)
        y = torch.clone(h)
    else:
        y = torch.squeeze(torch.matmul(A, torch.unsqueeze(h, 2)))
    snr_list = np.zeros([h.shape[0]])
    for i in range(h.shape[0]):
        if snr_scaling is not None:
            snr = np.random.choice(snrs, p=snr_scaling)
        else:
            snr = np.random.choice(snrs)
        snr_list[i] = snr
        #y[i] += 10 ** (-snr / 20) * crandn(*y[i].shape)
        y[i] = y[i] + 10 ** (-snr / 20) * np.sqrt(0.5) * (torch.randn_like(y[i]) + 1j * torch.randn_like(y[i]))
        if n_bits != np.inf:
            y[i] = quant_torch(y[i], n_bits, quantizer[snr][0], quantizer[snr][1], device=device)
    return y, snr_list


def get_observation_nbit_randSNR_torch_fast(h, snrs, A=None, n_bits=1, quantizer=None, snr_scaling=None, device='cpu'):
    if A is None:
        y = h.clone()
    else:
        y = torch.squeeze(torch.matmul(A, torch.unsqueeze(h, 2)))

    batch_size = y.shape[0]

    # Generate random SNRs for each sample in the batch
    if snr_scaling is not None:
        snr_probs = torch.tensor(snr_scaling, device=device)
        snr_indices = torch.multinomial(snr_probs, batch_size, replacement=True)
    else:
        snr_indices = torch.randint(0, len(snrs), (batch_size,), device=device)
    snr_list = torch.tensor([snrs[idx] for idx in snr_indices], device=device)

    # Add noise to the observation based on the SNR
    noise_power = 10 ** (-snr_list / 20) / np.sqrt(2)
    noise_real = torch.randn_like(y, device=device) * noise_power.view(-1, 1)
    noise_imag = torch.randn_like(y, device=device) * noise_power.view(-1, 1)
    y = y + noise_real + 1j * noise_imag

    if n_bits != np.inf:
        for i in range(batch_size):
            snr_idx = snr_indices[i]
            quant_params = quantizer[snrs[snr_idx]]
            y[i] = quant_torch(y[i], n_bits, quant_params[0], quant_params[1], device=device)
    return y, snr_list



def vec2matrix(inp, n_antennas, n_pilots):
    """Reshape vectorized pilot observations into matrix."""
    out = np.reshape(inp, [-1, n_antennas, n_pilots], 'F')
    return out


def get_observation_unquant(h, snr, A=None):
    if A is None:
        return h + 10 ** (-snr / 20) * crandn(*h.shape)
    else:
        y = np.squeeze(np.matmul(A, np.expand_dims(h, 2)))
        y += + 10 ** (-snr / 20) * crandn(*y.shape)
        return y


def get_pilot_matrix(n_antennas, n_pilots, n_bits, pilot_type='angle_amp', return_vector=False):
    if n_bits == np.inf or n_bits == 'inf':
        x = np.ones([n_pilots, 1])
    else:
        if pilot_type == 'angle':
            #dist = np.pi / (2 * 2**n_bits)
            dist = np.pi / 2
            #x = np.arange(0, dist, dist / n_pilots)
            x = np.linspace(0, dist, num=n_pilots, endpoint=False)
            x = np.exp(1j*x)
            x = np.expand_dims(x, 1)
        elif pilot_type == 'rand':
            x = np.random.randn(n_pilots, 1) + 1j*np.random.randn(n_pilots, 1)
            x *= np.sqrt(n_pilots) / np.linalg.norm(x) #normalize to fulfill power constraint
        elif pilot_type == 'angle_amp':
            #dist = np.pi / (2 * 2**n_bits)
            dist = np.pi / 2
            x = np.linspace(0, dist, num=n_pilots, endpoint=False)
            a = np.linspace(0.5, 1, num=n_pilots, endpoint=True)
            x = a * np.exp(1j*x)
            x *= np.sqrt(n_pilots) / np.linalg.norm(x)
            x = np.expand_dims(x, 1)
        elif pilot_type == 'ones':
            x = np.ones([n_pilots, 1])
        else:
            raise NotImplementedError(f'Pilot type {pilot_type} is not implemented!')
    if return_vector:
        return x
    else:
        X = np.kron(x, np.eye(n_antennas))
        return X


def rand_exp(left: float, right: float, shape: Tuple[int, ...]=(1,), seed=None):
    r"""For 0 < left < right draw uniformly between log(left) and log(right)
    and exponentiate the result.

    Note:
        This procedure is explained in
            "Random Search for Hyper-Parameter Optimization"
            by Bergstra, Bengio
    """
    if left <= 0:
        raise ValueError('left needs to be positive but is {}'.format(left))
    if right <= left:
        raise ValueError(f'right needs to be larger than left but we have left: {left} and right: {right}')
    rng = np.random.default_rng(seed)
    return np.exp(np.log(left) + rng.random(*shape) * (np.log(right) - np.log(left)))


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
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T
    else:
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / np.sqrt(covariances)
    return precisions_chol


def walk_directory(directory: str, startswith: str='', endswith: str=''):
    """Walk through a directory and return file names.

    Args:
        directory: Walk through this directory.
        startswith: Only find files starting with this string.
        endswith: Only find files ending with this string. This can for
            example be a file extension.

    Returns:
        A generator for the file names that have been found.

    Example:
        The following would print the name of all .csv files in my_dir:

        for filename in walk_directory('my_dir', endswith='.csv'):
            print(filename)
    """
    # encoding for platform independence
    dir_path = os.fsencode(directory)
    for filename in os.listdir(dir_path):
        f = os.fsdecode(filename)
        if f.startswith(startswith) and f.endswith(endswith):
            yield os.path.join(directory, f)


def sort(l: list):
    # windows-like sorting for linux (we want 9 < 10 not 10 < 9)
    return sorted(l, key=lambda s: int(s.split('/')[-1].split('_')[1]))


def sec2hours(seconds: float):
    """"Convert number of seconds to a string hh:mm:ss."""
    # hours
    h = seconds // 3600
    # remaining seconds
    r = seconds % 3600
    return '{:.0f}:{:02.0f}:{:02.0f}'.format(h, r // 60, r % 60)


def rand_geom(left: float, right: float, shape: Tuple[int, ...]=(1,), seed=None):
    r"""For 0 < left < right draw uniformly between log(left) and log(right)
    and exponentiate the result. Round the obtained numbers to their nearest
    integers.

    Note:
        This procedure is explained in
            "Random Search for Hyper-Parameter Optimization"
            by Bergstra, Bengio
    """
    rng = np.random.default_rng(seed)
    return np.round(rand_exp(left, right, shape, rng)).astype('int')


def print_dict(dict: dict, entries_per_row: int=1):
    """Print the keys and values of dictionary dict."""
    if entries_per_row < 1:
        raise ValueError(f'The number of entries per row needs to be >= 1 but is {entries_per_row}')
    for c, (key, value) in enumerate(dict.items()):
        if c % entries_per_row == 0 and c > 0:
            print()
        else:
            c > 0 and print(' | ', end='')
        print('{}: {}'.format(key, value), end='')
    print()


def cplx2real(vec: np.ndarray, axis=0):
    """
    Concatenate real and imaginary parts of vec along axis=axis.
    """
    return np.concatenate([vec.real, vec.imag], axis=axis)


def cplx2real_torch(vec, axis=0):
    """
    Concatenate real and imaginary parts of vec along axis=axis.
    """
    return torch.cat([torch.real(vec), torch.imag(vec)], dim=axis)



def dict_to_csv(dict: dict, filename: str='dict.csv'):
    """Write all (key, value) pairs of dictionary dict into a .csv."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        for key, val in dict.items():
            writer.writerow((key, val))


def cplx_1bit(inp):
    return 1 / np.sqrt(2) * (np.sign(np.real(inp)) + 1j* np.sign(np.imag(inp)))


def get_quantizer(snrs, n_bits, quantizer_type='uniform'):
    quantizer = dict()
    if n_bits == 'inf' or n_bits == np.inf or n_bits == 1:
        for snr in snrs:
            quantizer[snr] = (None, None, None)
    else:
        if quantizer_type == 'uniform':
            for snr in snrs:
                delta = get_uniform_quant_step(snr, n_bits)
                thresholds = np.zeros([int(2 ** n_bits - 1)])
                for nb in range(int((2 ** n_bits - 2) / 2)):
                    thresholds[nb] = -((2 ** n_bits - 2) / 2 - nb) * delta
                    thresholds[-nb - 1] = ((2 ** n_bits - 2) / 2 - nb) * delta
                quant_labels = np.zeros([int(2 ** n_bits)])
                for nb in range(int(2 ** n_bits - 1)):
                    quant_labels[nb] = thresholds[nb] - delta / 2
                quant_labels[-1] = thresholds[-1] + delta / 2
                input_var = None
                quantizer[snr] = (thresholds, quant_labels, input_var)
        elif quantizer_type == 'lloyd':
            input_quantizer = list()
            for snr in snrs:
                input_quantizer.append([snr, n_bits])
            n_processes = int(mp.cpu_count() / 2)  # int(mp.cpu_count() / 2 - 1)
            pool = mp.Pool(processes=n_processes)
            quantizer_res = pool.starmap(load_quantizer, input_quantizer)
            for iter_snr, snr in enumerate(snrs):
                dic = quantizer_res[iter_snr]
                quantizer[snr] = dic[snr]
        else:
            raise NotImplementedError(f'Quantizer type {quantizer_type} not implemented!')
    return quantizer


def get_quantizer_gauss(snrs, n_bits, quantizer_type='lloyd', params=None):
    # calculate quantization bounds
    quantizer = dict()
    if n_bits == 'inf' or n_bits == np.inf or n_bits == 1:
        for snr in snrs:
            quantizer[snr] = (None, None, None)
    else:
        if quantizer_type == 'uniform':
            for snr in snrs:
                delta = get_uniform_quant_step(snr, n_bits)
                thresholds = np.zeros([int(2 ** n_bits - 1)])
                for nb in range(int((2 ** n_bits - 2) / 2)):
                    thresholds[nb] = -((2 ** n_bits - 2) / 2 - nb) * delta
                    thresholds[-nb - 1] = ((2 ** n_bits - 2) / 2 - nb) * delta
                quant_labels = np.zeros([int(2 ** n_bits)])
                for nb in range(int(2 ** n_bits - 1)):
                    quant_labels[nb] = thresholds[nb] - delta / 2
                quant_labels[-1] = thresholds[-1] + delta / 2
                input_var = None
                quantizer[snr] = (thresholds, quant_labels, input_var)
        elif quantizer_type == 'lloyd':
            for snr in snrs:
                quantizer[snr] = load_quantizer(snr, n_bits)[snr]
        else:
            raise NotImplementedError(f'Quantizer type {quantizer_type} not implemented!')
    return quantizer


def check_random_state(seed):
    import numbers
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)




def mse(h_est, h):
    return np.sum(np.abs(h_est - h) ** 2) / h.size


def real2cplx(vec: np.ndarray, axis=0):
    """
    Assume vec consists of concatenated real and imaginary parts. Return the
    corresponding complex vector. Split along axis=axis.
    """
    re, im = np.split(vec, 2, axis=axis)
    return re + 1j * im

def real2cplx_torch(vec, axis=0):
    """
    Assume vec consists of concatenated real and imaginary parts. Return the
    corresponding complex vector. Split along axis=axis.
    """
    re, im = torch.split(vec, 2, dim=axis)
    return re + 1j * im


def crandn_vae(size: Tuple[int, ...], seed=None) -> np.ndarray:
    """Complex standard normal random numbers."""
    rng = np.random.default_rng(seed)
    return np.sqrt(0.5) * (rng.standard_normal(size) + 1j * rng.standard_normal(size))


def make_cplx_spd_matrix(dim):
    A = np.random.rand(dim, dim) + 1j * np.random.rand(dim, dim)
    U, _, Vt = linalg.svd(np.dot(A.T.conj(), A), check_finite=False)
    X = np.dot(np.dot(U, 1.0 + np.diag(np.random.rand(dim))), Vt)
    return X


def gauss_newt_solve(f, J, x0, tol=1e-5, maxits=100):
    """Gauss-Newton algorithm for solving nonlinear least squares problems.
    Parameters
    ----------
    sys : Dataset
        Class providing residuals() and jacobian() functions. The former should
        evaluate the residuals of a nonlinear system for a given set of
        parameters. The latter should evaluate the Jacobian matrix of said
        system for the same parameters.
    x0 : tuple, list or ndarray
        Initial guesses or starting estimates for the system.
    tol : float
        Tolerance threshold. The problem is considered solved when this value
        becomes smaller than the magnitude of the correction vector.
        Defaults to 1e-10.
    maxits : int
        Maximum number of iterations of the algorithm to perform.
        Defaults to 256.
    Return
    ------
    sol : ndarray
        Resultant values.
    its : int
        Number of iterations performed.
    Note
    ----
    Uses numpy.linalg.pinv() in place of similar functions from scipy, both
    because it was found to be faster and to eliminate the extra dependency.
    """
    dx = 1.0   # Correction vector
    xn = x0.copy()       # Approximation of solution

    i = 0
    while (i < maxits) and (np.abs(dx) > tol):
        if np.abs(xn) < 0.1:
            xn = x0 + 0.1 * np.random.randn()
            xn = np.clip(xn, 0.1, 10.0)
            #break
        elif np.abs(xn) > 10.0:
            xn = 1.0 + 0.1 * np.random.randn()
        # correction = pinv(jacobian) . residual vector

        dx = np.linalg.lstsq(np.expand_dims(J(xn),1), -f(xn), rcond=None)[0][0]
        #dx  = np.dot(np.linalg.pinv(J(xn)), -f(xn))
        xn += dx            # x_{n + 1} = x_n + dx_n
        i  += 1

    return xn, i