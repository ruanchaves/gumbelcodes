import os
import numpy as np
from gumbelcodes.utils.matutils import bitpack_uint8_matrix, bitunpack_uint8_matrix, matrix_bincount, safe_matrix_bincount
from loguru import logger
import codecs 

def make_npz(codes_path, codebook_path, target_path):
    codes = np.loadtxt(codes_path)
    codebook = np.load(codebook_path)

    codes = codes.astype(dtype=np.int64)
    max_value = codes.max()
    min_value = codes.min()
    
    codes = safe_matrix_bincount(max_value, codes)
    codes = bitpack_uint8_matrix(codes)

    array_length = max_value + 1
    codebook = codebook[0:array_length]

    np.savez_compressed(target_path, codes=codes, codebook=codebook)

def decompress_npz(npz_array_path):
    npz_array = np.load(npz_array_path)
    codes = npz_array['codes']
    codebook = npz_array['codebook']
    codes = bitunpack_uint8_matrix(codes)
    return codes, codebook

def restore_from_npz(npz_array_path):
    codes, codebook = decompress_npz(npz_array_path)
    new_codes = []
    for item in codes:
        new_codes.append(np.repeat(np.arange(len(item)), item))
    return np.array(new_codes), codebook

def embedding_from_bincounted(words_path, codes, codebook):
    codebook_transposed = codebook.T
    with codecs.open(words_path, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f):
            embedding = ( codebook_transposed * codes[idx] ).T
            embedding = np.sum(embedding, axis=0)
            embedding = ' '.join([str(x) for x in embedding])
            new_line = line.strip() + ' ' + embedding
            yield new_line

def embedding_from_file(words_path, codes, codebook):
    codebook_transposed = codebook.T
    with codecs.open(words_path, 'r', encoding='utf8') as f:
        for idx, line in enumerate(f):
            embedding = np.array([codebook[x] for x in codes[idx] ])
            embedding = np.sum(embedding, axis=0)
            embedding = ' '.join([str(x) for x in embedding])
            new_line = line.strip() + ' ' + embedding
            yield new_line

def expand_from_arrays(words_path, codes, codebook, target_path):
    open(target_path, 'w+').close()
    with codecs.open(target_path, 'a', encoding='utf8') as f:
        for item in embedding_from_bincounted(words_path, codes, codebook):
            print(item, file=f)

def expand_from_files(words_path, codes_path, codebook_path, target_path):
    codebook = np.load(codebook_path)
    codes = np.loadtxt(codes_path)
    codes = codes.astype(np.uint8)
    open(target_path, 'w+').close()
    with codecs.open(target_path, 'a', encoding='utf8') as f:
        for item in embedding_from_file(words_path, codes, codebook):
            print(item, file=f)