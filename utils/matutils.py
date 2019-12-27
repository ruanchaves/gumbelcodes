import numpy as np
from importlib.util import find_spec
from scipy.linalg import get_blas_funcs
from loguru import logger

def matrix_bincount(A):
    N = A.max()+1
    idx = A + (N*np.arange(A.shape[0]))[:,None]
    return N, np.bincount(idx.ravel(),minlength=N*A.shape[0]).reshape(-1,N)

def bitunpack_uint8_matrix(A):
    even = (A.astype(np.uint8) >> 4).T
    odd = (A.astype(np.uint8) & 0xF).T
    unpack = np.hstack([even, odd]).reshape(even.shape[0]*2, even.shape[1]).T
    return unpack.astype(np.uint8)

def bitpack_uint8_matrix(A):
    even = A.astype(np.uint8) << 4 
    odd = np.pad(A[..., 1:], ((0,0), (0,1))).astype(np.uint8)
    pack = even + odd
    return pack[:, ::2]

def safe_matrix_bincount(max_value, A, bitsize=16):
    assert(max_value < len(A[0]))
    max_rep, A = matrix_bincount(A)
    try:
        assert(max_value < bitsize)
    except:
        logger.debug("min_value = {0}, max_value = {1}".format(min_value, max_value))
        Exception('Out of range')
    try:
        assert(max_rep < bitsize)
    except:
        A[A > bitsize - 1] = bitsize - 1
        logger.debug("max_rep = {0}".format(max_rep))
    return A

def split_codebook(codebook):
    codebook_even = np.array(codebook[::2], dtype=np.float32)
    codebook_odd = np.array(codebook[1::2], dtype=np.float32)
    return np.array([codebook_even, codebook_odd], dtype=np.float32)

def blas(name, ndarray):
    """Helper for getting the appropriate BLAS function, using :func:`scipy.linalg.get_blas_funcs`.

    Parameters
    ----------
    name : str
        Name(s) of BLAS functions, without the type prefix.
    ndarray : numpy.ndarray
        Arrays can be given to determine optimal prefix of BLAS routines.

    Returns
    -------
    object
        BLAS function for the needed operation on the given data type.

    """
    return get_blas_funcs((name,), (ndarray,))[0]

blas_nrm2 = blas('nrm2', np.array([], dtype=float))
blas_scal = blas('scal', np.array([], dtype=float))

def unitvec(vec, norm='l2', return_norm=False):
    """Scale a vector to unit length.

    Parameters
    ----------
    vec : {numpy.ndarray, scipy.sparse, list of (int, float)}
        Input vector in any format
    norm : {'l1', 'l2', 'unique'}, optional
        Metric to normalize in.
    return_norm : bool, optional
        Return the length of vector `vec`, in addition to the normalized vector itself?

    Returns
    -------
    numpy.ndarray, scipy.sparse, list of (int, float)}
        Normalized vector in same format as `vec`.
    float
        Length of `vec` before normalization, if `return_norm` is set.

    Notes
    -----
    Zero-vector will be unchanged.

    """
    supported_norms = ('l1', 'l2', 'unique')
    if norm not in supported_norms:
        raise ValueError("'%s' is not a supported norm. Currently supported norms are %s." % (norm, supported_norms))

    if isinstance(vec, np.ndarray):
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        if norm == 'l2':
            if vec.size == 0:
                veclen = 0.0
            else:
                veclen = blas_nrm2(vec)
        if norm == 'unique':
            veclen = np.count_nonzero(vec)
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.integer):
                vec = vec.astype(np.float)
            if return_norm:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype), veclen
            else:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype)
        else:
            if return_norm:
                return vec, 1.0
            else:
                return vec

    try:
        first = next(iter(vec))  # is there at least one element?
    except StopIteration:
        if return_norm:
            return vec, 1.0
        else:
            return vec