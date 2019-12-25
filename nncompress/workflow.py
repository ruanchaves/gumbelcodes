from __future__ import absolute_import, division, print_function

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

from loguru import logger
from marisa_trie import Trie
from embed_compress import EmbeddingCompressor

def create_file(original_path, prefix=''):
    path = os.path.split(original_path)[0]
    fname = os.path.split(original_path)[1]
    fname = prefix + fname
    return os.path.join(path, fname)

def matrix_bincount(A):
    N = A.max()+1
    idx = A + (N*np.arange(A.shape[0]))[:,None]
    return N, np.bincount(idx.ravel(),minlength=N*A.shape[0]).reshape(-1,N)

def bitpack_uint8_matrix(A):
    even = A.astype(np.uint8) << 4 
    odd = np.pad(A[..., 1:], ((0,0), (0,1))).astype(np.uint8)
    pack = even + odd
    return pack[:, ::2]

class Pipeline(object):

    def __init__(self, 
                source_path=None, 
                dimension=None, 
                target_path=None, 
                dictionary_path=None, 
                trie_path=None, 
                codebook_prefix=None,
                logging=True):

        self.source_path = source_path
        self.dimension = dimension
        self.target_path = target_path
        self.dictionary_path = dictionary_path
        self.trie_path = trie_path
        self.codebook_prefix = codebook_prefix
        if logging:
            logger.debug("Source: {0} \n Target: {1}".format(self.source_path, self.target_path))

    def get_words(self, generate_trie=True, generate_dictionary=True):
        words = [x.strip().split()[0] for x in list(open(self.source_path, encoding="utf-8"))]

        if generate_dictionary:
            dictionary = { key : idx for idx, key in enumerate(words) }
            with open(self.dictionary_path,'w+') as f:
                json.dump(dictionary, f)
       
        if generate_trie:
            if generate_dictionary:
                del dictionary
            trie = Trie(words)
            trie.save(self.trie_path)

        return self

    def get_embeddings(self, ignore_error=True):
        lines = list(open(self.source_path, encoding="utf-8"))
        embed_matrix = np.zeros((len(lines), self.dimension), dtype='float32')
        logger.debug("Embedding shape: {0}".format(embed_matrix.shape))
        for i, line in enumerate(lines):
            parts = line.strip().split()
            try:
                vec = np.array(parts[1:], dtype='float32')
                embed_matrix[i] = vec
            except:
                logger.debug(line)
                logger.debug("Invalid embedding at line {0}".format(i))
                if not ignore_error:
                    raise LookupError("Invalid embedding")
        np.save(self.target_path, embed_matrix)    
        return self

        
    def train(self):
        matrix = np.load(self.target_path)
        compressor = EmbeddingCompressor(32, 16, self.codebook_prefix)
        compressor.train(matrix)
        distance = compressor.evaluate(matrix)
        logger.debug("Mean euclidean distance:", distance)
        compressor.export(matrix, self.codebook_prefix)
        return self

    def trim(self, prefix='trimmed_', suffix='.codes'):
        codes_path = self.codebook_prefix + suffix
        codes = np.loadtxt(codes_path)
        codes = codes.astype(dtype=np.int64)
        max_value = codes.max()

        assert((codes == codes[0]).all())
        assert(max_value < len(codes[0]))

        max_rep, codes = matrix_bincount(codes)

        assert(max_value < 16)
        assert(max_rep < 16)

        codes = bitpack_uint8_matrix(codes)

        np.save(create_file(codes_path, 'trimmed_'), codes)

        codebook_path = self.codebook_prefix + '.codebook.npy'
        codebook = np.load(codebook_path)
        array_length = max_value + 1
        codebook = codebook[0:array_length]
        codebook_even = np.array(codebook[::2], dtype=np.float32)
        codebook_odd = np.array(codebook[1::2], dtype=np.float32)
        codebook = np.array([codebook_even, codebook_odd], dtype=np.float32)

        np.save(create_file(codebook_path, 'trimmed_'), codebook)

if __name__ == '__main__':
    with open(sys.argv[1],'r') as f:
        settings = json.load(f)
    for entry in settings: 
        pipe = Pipeline(**entry)
        # pipe\
        #     .get_words(generate_dictionary=False)\
        #     .get_embeddings()\
        #     .train()\
        #     .trim()
        pipe\
            .trim()