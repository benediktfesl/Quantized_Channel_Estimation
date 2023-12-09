import numpy as np
from scipy.linalg import toeplitz, sqrtm
from utils import crandn


def crandn_toeplitz(t, rng=np.random.default_rng()):
    """
    Generate `x ~ N_C(0,T)` with Toeplitz cov. `T` that has `t` as its first row.
    """
    c = toeplitz(t)

    y = sqrtm(c) * crandn(len(t), rng=rng)
    return y


def apply_toeplitz(t, x):
    x2 = np.concatenate(x, np.zeros(x), axis=1)

    t2 = np.concatenate(np.conj(t), 0, np.flip(t, axis=0)[1:], axis=1)

    y = np.fft.ifft(np.fft.fft(t2, axis=0) * np.fft.fft(x2, axis=0), axis=0)

    y = y[0:x.size]
    return y


def best_block_circulant_approximation(t):
    """
    If `t` is the length `n` generating vector of a Hermitian Toeplitz matrix `T`, then
    `c` is the length `n` generating vector of a Hermitian circulant matrix `C`, which
    is the best approximation of `T`.
    """
    if np.ndim(t) is 1:
        n = t.size

        lins = np.arange(n, -n+1, -2) / n

        t1 = t
        t2 = np.hstack((np.zeros(1, dtype=np.complex64), np.conj(np.flip(t[1:], axis=0))))

        c = 0.5 * n * np.real(np.fft.ifft(lins * (t1 - t2) + t1 + t2, axis=0))
        return c

    elif np.ndim(t) is 2:
        n = t.size[0]
        m = t.size[1]

        lins = np.arange(n, n + 1, -2) / n

        t1 = t
        t2 = np.concatenate(np.zeros((1, m), dtype=np.complex128), np.conj(np.flip(t[1:, :], axis=0)), axis=0)

        c = 0.5 * n * np.real(np.fft.ifft(lins * (t1 - t2) + t1 + t2, axis=0))
        return c


def best_circulant_approximation(t):
    """
    If `t` is the length `n` generating vector of a Hermitian Toeplitz matrix `T`, then
    `c` is the length `n` generating vector of a Hermitian circulant matrix `C`, which
    is the best approximation of `T`.
    """
    if np.ndim(t) is 1:
        n = t.size

        lins = np.arange(n, -n+1, -2) / n

        t1 = t
        t2 = np.hstack((np.zeros(1, dtype=np.complex64), np.conj(np.flip(t[1:], axis=0))))

        c = 0.5 * n * np.real(np.fft.ifft(lins * (t1 - t2) + t1 + t2, axis=0))
        return c

    elif np.ndim(t) is 2:
        n = t.size[0]
        m = t.size[1]

        lins = np.arange(n, n + 1, -2) / n

        t1 = t
        t2 = np.concatenate(np.zeros((1, m), dtype=np.complex128), np.conj(np.flip(t[1:, :], axis=0)), axis=0)

        c = 0.5 * n * np.real(np.fft.ifft(lins * (t1 - t2) + t1 + t2, axis=0))
        return c


def vec2mat(vec, ndim1, ndim2):
    # reshaping vectorized form of (batches,1,ndim1*ndim2) to matrix of shape (batches,1,ndim1,ndim2)
    s = vec.shape
    if len(s) == 1:
        mat = vec.reshape((ndim2, ndim1))
        mat = np.transpose(mat, axes=[1, 0])
    elif len(s) == 2:
        mat = vec.reshape((-1, ndim2, ndim1))
        mat = np.transpose(mat, axes=[0, 2, 1])
    else:
        mat = vec.reshape((-1,s[1], ndim2, ndim1))
        mat = np.transpose(mat, axes=[0, 1, 3, 2])
    return mat


def mat2vec(mat):
    # reshaping (batches,1,ndim1,ndim2) to vector of shape (batches,1,ndim1*ndim2)
    s = mat.shape
    if len(s) == 2:
        mat = np.transpose(mat, axes=[1, 0])
        vec = mat.reshape((s[0]*s[1]))
    elif len(s) == 3:
        mat = np.transpose(mat, axes=[0, 2, 1])
        vec = mat.reshape((-1, s[1]*s[2]))
    else:
        mat = np.transpose(mat, axes=[0, 1, 3, 2])
        vec = mat.reshape((-1, s[1], s[-1]*s[-2]))
    return vec
