from __future__ import absolute_import, division, print_function

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

from loguru import logger
from marisa_trie import Trie
from embed_compress import EmbeddingCompressor

def encode(x,y):
    return (x << 4) + y

def create_file(original_path, prefix=''):
    path = os.path.split(original_path)[0]
    fname = os.path.split(original_path)[1]
    fname = prefix + fname
    return os.path.join(path, fname)


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
        max_value = np.max(codes.flatten())

        assert(not [x for x in codes if len(x) != len(codes[0])])
        assert(max_value < len(codes[0]) )

        new_codes = np.array([ np.bincount(x, minlength=(max_value+1)) for x in codes ], dtype=np.uint8)
        max_rep = np.max(new_codes.flatten())

        assert(max_value < 16)
        assert(max_rep < 16)

        new_codes = [ [ x[idx:idx+2] for idx in range(0,len(x), 2) ] for x in new_codes ]
        new_codes = [ np.array([ encode(*y) for y in x ],dtype=np.uint8) for x in new_codes ]
        new_codes = np.array(new_codes, dtype=np.uint8)

        np.save(create_file(codes_path, 'trimmed_'), new_codes)


        codebook_path = self.codebook_prefix + '.codebook.npy'
        codebook = np.load(codebook_path)
        array_length = max_value + 1
        codebook = codebook[0:array_length]
        codebook_odd = np.array(codebook[::2], dtype=np.float32)
        codebook_even = np.array(codebook[1::2], dtype=np.float32)
        codebook = np.array([codebook_odd, codebook_even], dtype=np.float32)

        np.save(create_file(codebook_path, 'trimmed_'), codebook)

if __name__ == '__main__':
    with open(sys.argv[1],'r') as f:
        settings = json.load(f)
    for entry in settings: 
        pipe = Pipeline(**entry)
        pipe\
            .get_words(generate_dictionary=False)\
            .get_embeddings()\
            .train()\
            .trim()
