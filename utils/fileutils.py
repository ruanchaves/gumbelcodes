import os
import numpy as np
from gumbelcodes.utils.matutils import bitpack_uint8_matrix, matrix_bincount
from loguru import logger

def compress(codes_path, codebook_path, target_path, logging=True):
    codes = np.loadtxt(codes_path)
    codebook = np.load(codebook_path)

    codes = codes.astype(dtype=np.int64)
    max_value = codes.max()
    min_value = codes.min()

    assert(max_value < len(codes[0]))

    max_rep, codes = matrix_bincount(codes)

    try:
        assert(max_value < 16)
    except:
        if logging:
            logger.debug("min_value = {1}, max_value = {2}".format(
                min_value, max_value))
        raise Exception("This embedding cannot be compressed.")

    try:
        assert(max_rep < 16)
    except:
        codes[codes > 15] = 15
        if logging:
            logger.debug(
                "{0} : max_rep = {1} - rounded down to 15".format(self.codebook_prefix, max_rep))

    codes = bitpack_uint8_matrix(codes)

    array_length = max_value + 1
    codebook = codebook[0:array_length]

    np.savez_compressed(target_path, codes=codes, codebook=codebook)

def decompress(npz_array_path):
    npz_array = np.load(npz_array_path)
    codes = npz_array['codes']
    codebook = npz_array['codebook']
    new_codes = []
    for item in codes:
        new_codes.append(np.repeat(np.arange(len(item)), item))
    return np.array(new_codes), codebook

def embedding_file_generator(words_path, codes, codebook, target_path):
    codebook_transposed = codebook.T
    with open(words_path,'r') as f:
        for idx, line in enumerate(f):
            embedding = ( codebook_transposed * codes[idx] ).T
            embedding = ' '.join(str(x) for x in embedding)
            new_line = line.strip() + ' ' + embedding
            yield new_line

def expand(words_path, codes, codebook, target_path):
    open(target_path, 'w+').close()
    with open(target_path, 'a') as f:
        for item in embedding_file_generator(words_path, codes, codebook, target_path):
            print(item, file=f)